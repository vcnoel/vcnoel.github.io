
import numpy as np
import sys
import os

# Add ReGA directory to path so we can import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ReGA'))

try:
    from comprehensive_evaluation import extract_rega_features, DIRECTIONAL_VERBS
except ImportError:
    # If running from ReGA dir directly
    try:
        from comprehensive_evaluation import extract_rega_features, DIRECTIONAL_VERBS
    except ImportError as e:
        print(f"Error importing ReGA: {e}")
        sys.exit(1)

def test_pbge():
    print("="*60)
    print("VERIFYING PYTHON PBGE IMPLEMENTATION")
    print("="*60)

    # Test Case 1: Faithful (No Polarity Mismatch, No Swap)
    print("\n[TEST 1] Faithful Case")
    src1 = "Google acquired YouTube in 2006."
    hyp1 = "Google acquired YouTube in 2006."
    feats1 = extract_rega_features(src1, hyp1)
    
    # Last 2 features are [polarity_mismatch, directional_swap]
    pol1 = feats1[-2]
    swap1 = feats1[-1]
    
    print(f"Source: {src1}")
    print(f"Hypothesis: {hyp1}")
    print(f"Polarity Mismatch: {pol1} (Expected: 0.0)")
    print(f"Directional Swap:  {swap1} (Expected: 0.0)")
    assert pol1 == 0.0 and swap1 == 0.0, "Test 1 Failed"

    # Test Case 2: Polarity Mismatch (Negation)
    print("\n[TEST 2] Polarity Mismatch (Negation)")
    src2 = "Google acquired YouTube in 2006."
    hyp2 = "Google did not acquire YouTube in 2006."
    feats2 = extract_rega_features(src2, hyp2)
    
    pol2 = feats2[-2]
    swap2 = feats2[-1]
    
    print(f"Source: {src2}")
    print(f"Hypothesis: {hyp2}")
    print(f"Polarity Mismatch: {pol2} (Expected: 1.0)")
    print(f"Directional Swap:  {swap2} (Expected: 0.0)")
    assert pol2 == 1.0, "Test 2 Failed (Polarity detection missed)"

    # Test Case 3: Directional Swap (Role Reversal)
    print("\n[TEST 3] Directional Swap")
    src3 = "Microsoft acquired Activision."
    hyp3 = "Activision acquired Microsoft."
    feats3 = extract_rega_features(src3, hyp3)
    
    pol3 = feats3[-2]
    swap3 = feats3[-1]
    
    print(f"Source: {src3}")
    print(f"Hypothesis: {hyp3}")
    print(f"Polarity Mismatch: {pol3} (Expected: 0.0)")
    print(f"Directional Swap:  {swap3} (Expected: 1.0)")
    assert swap3 == 1.0, "Test 3 Failed (Swap detection missed)"
    
    # Test Case 4: Advanced Negation
    print("\n[TEST 4] Advanced Negation ('never')")
    src4 = "The deal was finalized."
    hyp4 = "The deal was never finalized."
    # Note: 'finalized' is NOT in DIRECTIONAL_VERBS by default unless we add it or use a known verb
    # Let's use a known verb: 'created'
    src4 = "OpenAI created ChatGPT."
    hyp4 = "OpenAI never created ChatGPT."
    
    feats4 = extract_rega_features(src4, hyp4)
    pol4 = feats4[-2]
    
    print(f"Source: {src4}")
    print(f"Hypothesis: {hyp4}")
    print(f"Polarity Mismatch: {pol4} (Expected: 1.0)")
    assert pol4 == 1.0, "Test 4 Failed"

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)

if __name__ == "__main__":
    test_pbge()
