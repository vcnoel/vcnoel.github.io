"""
COMPREHENSIVE EVALUATION WITH 5000+ SAMPLES
Tests ReGA vs LLM judges on each hallucination category
"""

import numpy as np
import json
import time
import urllib.request
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

NVIDIA_API_KEY = "NVIDIA_API_KEY"  # ← ADD YOUR KEY!
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Models to test
MODELS_TO_TEST = [
    "meta/llama-3.1-70b-instruct",      # Fast and good
    "google/gemma-2-27b-it",            # Alternative
]

# Sampling strategy (to reduce API costs)
SAMPLES_PER_CATEGORY_FOR_LLMS = 100  # Test LLMs on 100 samples per category
FULL_TEST_FOR_REGA = True  # Test ReGA on full dataset

# ============================================================================
# EMBEDDINGS
# ============================================================================

def load_embedder():
    """Load sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Loading sentence-transformers (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  Model loaded successfully!\n")
        return model, True
    except:
        print("⚠ sentence-transformers not available")
        return None, False

EMBEDDER, USE_REAL_EMBEDDINGS = load_embedder()

def encode_text(text):
    """Encode text"""
    if USE_REAL_EMBEDDINGS:
        return EMBEDDER.encode(text)
    else:
        np.random.seed(hash(text) % 2**31)
        emb = np.random.randn(384) * 0.1
        for word in text.lower().split():
            np.random.seed(hash(word) % 2**31)
            emb += np.random.randn(384) * 0.3
        return emb / (np.linalg.norm(emb) + 1e-8)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

DIRECTIONAL_VERBS = [
    'acquired', 'acquire', 'bought', 'buy', 'sold', 'sell', 'purchased', 'purchase',
    'defeated', 'beat', 'won against', 'win against', 'lost to', 'lose to',
    'sued', 'sue', 'filed against', 'file against',
    'invented', 'invent', 'created', 'create', 'founded', 'built', 'build',
    'wrote', 'write', 'authored', 'author',
    'killed', 'kill', 'murdered', 'murder', 'attacked', 'attack'
]

NEGATION_MARKERS = ["not", "did not", "didn't", "never", "refused to", "failed to", "was not", "is not"]

def extract_triplet(text, verb):
    """
    Extract (Subject, Verb, Object, Polarity) triplet using regex-like splitting.
    Polarity: 1 (Positive), -1 (Negative)
    """
    # Normalize
    text_norm = text.lower().replace('.', '').replace(',', '')
    parts = text_norm.split(verb)
    
    if len(parts) < 2:
        return None
        
    subject = parts[0].strip()
    object = parts[1].strip()
    polarity = 1
    
    # Check negation
    for marker in NEGATION_MARKERS:
        if subject.endswith(marker):
            polarity = -1
            subject = subject[:-len(marker)].strip()
            break
            
    return {'s': subject, 'o': object, 'v': verb, 'p': polarity}

def extract_rega_features(source_text, hypothesis_text):
    """Extract ReGA features"""
    # Split into sentences/phrases
    source_sents = [s.strip() for s in source_text.replace('.', '|').split('|') if s.strip()]
    hyp_sents = [s.strip() for s in hypothesis_text.replace('.', '|').split('|') if s.strip()]
    
    # Encode
    src_embs = np.array([encode_text(s) for s in source_sents])
    hyp_embs = np.array([encode_text(s) for s in hyp_sents])
    
    features = []
    
    # Graph-level
    mean_src = np.mean(src_embs, axis=0)
    mean_hyp = np.mean(hyp_embs, axis=0)
    features.append(np.linalg.norm(mean_src - mean_hyp))
    features.append(1 - cosine_similarity([mean_src], [mean_hyp])[0,0])
    
    # Entity-level
    n = min(len(src_embs), len(hyp_embs))
    if n > 0:
        costs = [1 - cosine_similarity([src_embs[i]], [hyp_embs[i]])[0,0] for i in range(n)]
        features.extend([np.mean(costs), np.max(costs), np.min(costs), np.std(costs)])
    else:
        features.extend([0, 0, 0, 0])
    
    # SWAP INDICATOR
    cost_matrix = 1 - cosine_similarity(src_embs, hyp_embs)
    diag = np.diag(cost_matrix).mean() if n > 0 else 0
    mins = cost_matrix.min(axis=1).mean()
    features.append(diag - mins)
    
    features.extend([cost_matrix.mean(), cost_matrix.std()])
    
    # ============================================================================
    # PBGE (Pattern-Based Graph Extraction) - "Kill Switch" Features
    # ============================================================================
    polarity_mismatch = 0.0
    directional_swap = 0.0
    
    src_lower = source_text.lower()
    hyp_lower = hypothesis_text.lower()
    
    for verb in DIRECTIONAL_VERBS:
        if verb in src_lower and verb in hyp_lower:
            src_triplet = extract_triplet(source_text, verb)
            hyp_triplet = extract_triplet(hypothesis_text, verb)
            
            if src_triplet and hyp_triplet:
                # 1. Polarity Mismatch
                if src_triplet['p'] != hyp_triplet['p']:
                    polarity_mismatch = 1.0
                
                # 2. Directional Swap
                # Check if subject/object are reversed (fuzzy match)
                s_s, s_o = src_triplet['s'], src_triplet['o']
                h_s, h_o = hyp_triplet['s'], hyp_triplet['o']
                
                # Swap logic: Source Subject matches Hyp Object OR Source Object matches Hyp Subject
                sub_swapped = (h_s in s_o) or (s_o in h_s)
                obj_swapped = (h_o in s_s) or (s_s in h_o)
                
                if sub_swapped or obj_swapped:
                    directional_swap = 1.0
            
            if polarity_mismatch or directional_swap:
                break # Stop at first detection
                
    features.append(polarity_mismatch)
    features.append(directional_swap)
    
    return np.array(features, dtype=np.float32)

# ============================================================================
# NVIDIA NIM API
# ============================================================================

def call_nvidia_nim(model_name, source, hypothesis, api_key):
    """Call NVIDIA NIM API"""
    
    prompt = f"""You are evaluating whether a hypothesis contains hallucinations relative to source text.

SOURCE:
{source}

HYPOTHESIS:
{hypothesis}

Question: Does the hypothesis contain any factual errors, role reversals, or contradictions compared to the source?

Answer with EXACTLY this format:
ANSWER: YES or NO
CONFIDENCE: 0.0 to 1.0
REASON: brief explanation"""

    url = f"{NVIDIA_BASE_URL}/chat/completions"
    
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 200
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result['choices'][0]['message']['content']
            
            is_hallucination = 'YES' in text.split('ANSWER:')[1].split('\n')[0].upper()
            
            try:
                conf_line = text.split('CONFIDENCE:')[1].split('\n')[0]
                confidence = float(''.join(c for c in conf_line if c.isdigit() or c == '.'))
                confidence = min(max(confidence, 0.0), 1.0)
            except:
                confidence = 0.7
            
            return is_hallucination, confidence, True, None
    
    except Exception as e:
        return False, 0.5, False, str(e)[:100]

# ============================================================================
# EVALUATION
# ============================================================================

def train_rega(train_data):
    """Train ReGA"""
    print("\n" + "="*80)
    print("TRAINING ReGA")
    print("="*80 + "\n")
    
    print(f"Extracting features from {len(train_data)} samples...")
    X_train = []
    y_train = []
    
    for i, case in enumerate(train_data):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(train_data)}")
        features = extract_rega_features(case['source'], case['hypothesis'])
        X_train.append(features)
        y_train.append(case['label'])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print("\nTraining logistic regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print(f"✓ ReGA trained\n")
    
    return model, scaler

def evaluate_rega(model, scaler, test_data):
    """Evaluate ReGA"""
    print(f"\nTesting ReGA on {len(test_data)} samples...")
    
    X_test = []
    for i, case in enumerate(test_data):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}")
        features = extract_rega_features(case['source'], case['hypothesis'])
        X_test.append(features)
    
    X_test = np.array(X_test)
    X_test_scaled = scaler.transform(X_test)
    
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    return preds, probs

def evaluate_llm_subset(model_name, test_data, api_key, max_samples=100):
    """Evaluate LLM on subset"""
    print(f"\nTesting {model_name} on {min(max_samples, len(test_data))} samples...")
    
    # Sample data
    sample_data = random.sample(test_data, min(max_samples, len(test_data)))
    
    predictions = []
    confidences = []
    times = []
    
    for i, case in enumerate(sample_data):
        print(f"  Case {i+1}/{len(sample_data)}...", end='', flush=True)
        
        start = time.time()
        is_hall, conf, success, error = call_nvidia_nim(
            model_name, case['source'], case['hypothesis'], api_key
        )
        elapsed = time.time() - start
        
        if success:
            predictions.append(1 if is_hall else 0)
            confidences.append(conf if is_hall else 1 - conf)
            times.append(elapsed)
            print(f" {elapsed:.2f}s ✓")
        else:
            predictions.append(0)
            confidences.append(0.5)
            times.append(0)
            print(f" ERROR")
        
        time.sleep(0.5)  # Rate limit
    
    return sample_data, np.array(predictions), np.array(confidences), times

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE EVALUATION: 5000+ Samples")
    print("="*80 + "\n")
    
    # Check API key
    use_api = NVIDIA_API_KEY != "YOUR_NVIDIA_API_KEY_HERE"
    if not use_api:
        print("⚠️ No API key configured - ReGA-only evaluation\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset_path = Path('comprehensive_hallucination_dataset.json')
    
    if not dataset_path.exists():
        print("❌ Dataset not found! Run generate_comprehensive_dataset.py first\n")
        return
    
    with open(dataset_path, 'r') as f:
        full_dataset = json.load(f)
    
    print(f"✓ Loaded {len(full_dataset)} samples\n")
    
    # Split by category
    by_category = defaultdict(list)
    for item in full_dataset:
        by_category[item['category']].append(item)
    
    print("Samples per category:")
    for cat, data in sorted(by_category.items()):
        print(f"  {cat}: {len(data)}")
    print()
    
    # Train/test split
    train_data = []
    test_data = []
    
    for cat, data in by_category.items():
        random.shuffle(data)
        split_idx = int(0.7 * len(data))
        train_data.extend(data[:split_idx])
        test_data.extend(data[split_idx:])
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    print(f"Train: {len(train_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    # Train ReGA
    rega_model, rega_scaler = train_rega(train_data)
    
    # Evaluate ReGA on FULL test set
    print("\n" + "="*80)
    print("EVALUATING ReGA ON FULL TEST SET")
    print("="*80)
    
    rega_start = time.time()
    rega_preds, rega_probs = evaluate_rega(rega_model, rega_scaler, test_data)
    rega_time = time.time() - rega_start
    
    y_true = np.array([d['label'] for d in test_data])
    
    # Overall metrics
    rega_acc = accuracy_score(y_true, rega_preds)
    rega_auc = roc_auc_score(y_true, rega_probs)
    rega_f1 = f1_score(y_true, rega_preds)
    
    print(f"\n✓ ReGA Overall Results:")
    print(f"  Accuracy: {rega_acc:.3f}")
    print(f"  AUC: {rega_auc:.3f}")
    print(f"  F1: {rega_f1:.3f}")
    print(f"  Time: {rega_time:.2f}s ({rega_time/len(test_data)*1000:.2f}ms per sample)")
    
    # Per-category results
    print("\n" + "="*80)
    print("ReGA PERFORMANCE BY CATEGORY")
    print("="*80 + "\n")
    
    category_results = {}
    
    for cat in sorted(by_category.keys()):
        cat_indices = [i for i, d in enumerate(test_data) if d['category'] == cat]
        if not cat_indices:
            continue
        
        cat_y_true = y_true[cat_indices]
        cat_preds = rega_preds[cat_indices]
        cat_probs = rega_probs[cat_indices]
        
        cat_acc = accuracy_score(cat_y_true, cat_preds)
        cat_auc = roc_auc_score(cat_y_true, cat_probs)
        cat_f1 = f1_score(cat_y_true, cat_preds)
        
        precision, recall, _, _ = precision_recall_fscore_support(
            cat_y_true, cat_preds, average='binary', zero_division=0
        )
        
        category_results[cat] = {
            'accuracy': float(cat_acc),
            'auc': float(cat_auc),
            'f1': float(cat_f1),
            'precision': float(precision),
            'recall': float(recall),
            'n_samples': len(cat_indices)
        }
        
        print(f"{cat}:")
        print(f"  Accuracy: {cat_acc:.3f}")
        print(f"  AUC: {cat_auc:.3f}")
        print(f"  F1: {cat_f1:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  Samples: {len(cat_indices)}\n")
    
    # Evaluate LLMs on subset
    llm_results = {}
    
    if use_api:
        print("\n" + "="*80)
        print(f"EVALUATING LLMs ON SUBSET ({SAMPLES_PER_CATEGORY_FOR_LLMS} per category)")
        print("="*80)
        
        # Sample from each category
        llm_test_data = []
        for cat, data in by_category.items():
            cat_test = [d for d in test_data if d['category'] == cat]
            sample_size = min(SAMPLES_PER_CATEGORY_FOR_LLMS, len(cat_test))
            llm_test_data.extend(random.sample(cat_test, sample_size))
        
        print(f"\nLLM test set: {len(llm_test_data)} samples\n")
        
        for model_name in MODELS_TO_TEST:
            print(f"\n{'='*80}")
            print(f"MODEL: {model_name}")
            print("="*80)
            
            try:
                sample_data, preds, confs, times = evaluate_llm_subset(
                    model_name, llm_test_data, NVIDIA_API_KEY, len(llm_test_data)
                )
                
                sample_y_true = np.array([d['label'] for d in sample_data])
                
                acc = accuracy_score(sample_y_true, preds)
                auc = roc_auc_score(sample_y_true, confs)
                f1 = f1_score(sample_y_true, preds)
                avg_time = np.mean([t for t in times if t > 0])
                
                print(f"\n✓ {model_name} Results:")
                print(f"  Accuracy: {acc:.3f}")
                print(f"  AUC: {auc:.3f}")
                print(f"  F1: {f1:.3f}")
                print(f"  Avg time: {avg_time:.2f}s per sample")
                
                # Per-category for LLM
                print(f"\n  Per-category:")
                llm_cat_results = {}
                for cat in sorted(by_category.keys()):
                    cat_sample_indices = [i for i, d in enumerate(sample_data) if d['category'] == cat]
                    if len(cat_sample_indices) < 10:
                        continue
                    
                    cat_y = sample_y_true[cat_sample_indices]
                    cat_p = preds[cat_sample_indices]
                    cat_acc = accuracy_score(cat_y, cat_p)
                    
                    llm_cat_results[cat] = float(cat_acc)
                    print(f"    {cat}: {cat_acc:.3f} (n={len(cat_sample_indices)})")
                
                llm_results[model_name] = {
                    'overall': {
                        'accuracy': float(acc),
                        'auc': float(auc),
                        'f1': float(f1),
                        'time_per_sample': float(avg_time)
                    },
                    'by_category': llm_cat_results
                }
                
            except Exception as e:
                print(f"❌ Failed: {e}")
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80 + "\n")
    
    print(f"{'Category':<35} {'ReGA Acc':<12} {'Best LLM':<12} {'Gap':<10}")
    print("-"*80)
    
    for cat in sorted(category_results.keys()):
        rega_cat_acc = category_results[cat]['accuracy']
        
        # Find best LLM for this category
        best_llm_acc = 0.0
        for model_name, llm_data in llm_results.items():
            if cat in llm_data['by_category']:
                best_llm_acc = max(best_llm_acc, llm_data['by_category'][cat])
        
        gap = best_llm_acc - rega_cat_acc if best_llm_acc > 0 else 0.0
        
        status = "✓✓✓" if gap < 0.05 else "✓✓" if gap < 0.10 else "✓" if gap < 0.15 else "✗"
        
        llm_str = f"{best_llm_acc:.3f}" if best_llm_acc > 0 else "N/A"
        gap_str = f"{gap:+.3f}" if best_llm_acc > 0 else "N/A"
        
        print(f"{cat:<35} {rega_cat_acc:<12.3f} {llm_str:<12} {gap_str:<10} {status}")
    
    # Save results
    results = {
        'rega_overall': {
            'accuracy': float(rega_acc),
            'auc': float(rega_auc),
            'f1': float(rega_f1),
            'time_total': float(rega_time),
            'time_per_sample': float(rega_time / len(test_data))
        },
        'rega_by_category': category_results,
        'llm_judges': llm_results,
        'test_set_size': len(test_data),
        'used_real_embeddings': USE_REAL_EMBEDDINGS
    }
    
    output_path = Path('/mnt/user-data/outputs') / 'comprehensive_evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to: comprehensive_evaluation_results.json")
    
    # VERDICT
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80 + "\n")
    
    # Find best and worst categories
    best_cat = max(category_results.items(), key=lambda x: x[1]['accuracy'])
    worst_cat = min(category_results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"Best category: {best_cat[0]}")
    print(f"  ReGA accuracy: {best_cat[1]['accuracy']:.1%}")
    print(f"  AUC: {best_cat[1]['auc']:.3f}")
    
    print(f"\nWorst category: {worst_cat[0]}")
    print(f"  ReGA accuracy: {worst_cat[1]['accuracy']:.1%}")
    print(f"  AUC: {worst_cat[1]['auc']:.3f}")
    
    # Count excellent categories (>= 90% accuracy)
    excellent = [cat for cat, res in category_results.items() if res['accuracy'] >= 0.90]
    good = [cat for cat, res in category_results.items() if 0.80 <= res['accuracy'] < 0.90]
    acceptable = [cat for cat, res in category_results.items() if 0.70 <= res['accuracy'] < 0.80]
    poor = [cat for cat, res in category_results.items() if res['accuracy'] < 0.70]
    
    print(f"\nCategory performance:")
    print(f"  Excellent (≥90%): {len(excellent)} - {', '.join(excellent)}")
    print(f"  Good (80-90%): {len(good)} - {', '.join(good)}")
    print(f"  Acceptable (70-80%): {len(acceptable)} - {', '.join(acceptable)}")
    print(f"  Poor (<70%): {len(poor)} - {', '.join(poor)}")
    
    if len(excellent) >= 2:
        print("\n🎉 STRONG RESULTS! ReGA excels on multiple categories!")
        print("   Paper angle: Domain-specific efficacy analysis")
    elif len(excellent) >= 1 and len(good) >= 1:
        print("\n✅ GOOD RESULTS! ReGA shows clear strengths!")
        print("   Paper angle: Selective deployment for specific hallucination types")
    elif len(good) + len(excellent) >= 3:
        print("\n⚖️ MODERATE RESULTS! ReGA has potential!")
        print("   Paper angle: Efficiency-accuracy tradeoff analysis")
    else:
        print("\n⚠️ MIXED RESULTS. Need improvement or different framing.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()
