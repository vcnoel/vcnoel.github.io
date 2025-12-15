#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HalluGraph v4 — Publication-grade CLI
Graph-alignment metrics for hallucination quantification
- Multi-LLM validation (Llama / Mistral; easily extensible)
- Edge-aware Relation Preservation (RP=0 when |Ea|=0)
- Numeric & unit normalization for robust AE/RP
- Statistical rigor: Wilcoxon + bootstrap CI + Cliff's δ
- Calibration exports, confusion matrices, ROC curves
- spaCy-only ablation
- Reproducible: seed + run config dump
"""

import os
import re
import json
import time
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine
from scipy import stats

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# -------------------------------
# Safe, optional imports for LLM
# -------------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Allow running spaCy-only mode without the package


# =============================================================================
# DATA
# =============================================================================

CONTEXTS = {
    "coral_biology": """
The Global Climate Research Institute published a comprehensive study in 2023 
examining the effects of ocean acidification on coral reef ecosystems in the 
Pacific Ocean. The research team, led by Dr. Marina Kowalski, collected data 
from 150 reef sites across a 5-year period from 2018 to 2023. 

Key findings include:
- Ocean pH levels decreased by 0.08 units on average across monitored sites
- Coral bleaching events increased by 34% compared to the 2010-2015 baseline
- Species diversity in affected reefs declined by 23% over the study period
- The Great Barrier Reef experienced the most severe impact with a 41% decline
- Three coral species were newly classified as critically endangered: 
  Acropora tenuis, Montipora capitata, and Pocillopora damicornis
- Temperature spikes above 30°C for more than 4 weeks triggered mass bleaching
- The research utilized underwater drones, satellite imagery, and manual surveys
- Funding was provided by the National Oceanic Foundation with a budget of $12.4M
""",
    "medical": """
A landmark clinical trial conducted at Johns Hopkins University from 2020 to 2024
investigated the efficacy of a novel immunotherapy treatment for melanoma patients.
The study, directed by Dr. Sarah Chen, enrolled 420 participants across 15 medical
centers in North America.

Results demonstrated:
- Overall survival rate improved to 68% at 5 years, compared to 42% with standard chemotherapy
- Complete response was observed in 28% of patients receiving the immunotherapy
- Partial response occurred in 45% of cases
- The treatment involved monthly infusions of pembrolizumab at 200mg dosage
- Common side effects included fatigue (65% of patients), rash (34%), and hypothyroidism (18%)
- Median progression-free survival was 14.2 months versus 7.8 months in the control group
- The therapy works by blocking PD-1 proteins on T-cells, enhancing immune response
- Total trial cost was $47 million, funded by the National Cancer Institute
""",
    "technology": """
TechCorp announced in Q3 2024 the release of their new quantum computing processor,
the QuantumX-5000, representing a significant breakthrough in computational power.
The project was led by Chief Scientist Dr. James Liu over a 3-year development cycle.

Technical specifications:
- The processor features 5000 qubits with 99.7% fidelity
- Operating temperature maintained at 15 millikelvin using dilution refrigeration
- Gate operation time reduced to 20 nanoseconds, 5x faster than previous generation
- Error correction implemented using surface code topology
- The system achieved quantum supremacy on specific optimization problems
- Benchmark tests showed 10^6 speedup over classical supercomputers for certain algorithms
- Power consumption is 25 kilowatts during active computation
- Development cost totaled $180 million with $120M from venture capital
- Applications target drug discovery, cryptography, and financial modeling
""",
    "economics": """
The Federal Reserve released its comprehensive economic report for fiscal year 2024,
analyzing inflation trends and monetary policy impacts. The report, authored by
Chair Jerome Powell's team, synthesized data from all 12 regional Federal Reserve Banks.

Key economic indicators:
- GDP growth rate was 2.8% annually, exceeding Q1 projections of 2.1%
- Core inflation (excluding food and energy) declined to 3.2% from 5.4% in 2023
- Unemployment rate stabilized at 4.1%, with 6.8 million job openings
- Federal funds rate was raised to 5.50% through six consecutive 25-basis-point increases
- Consumer Price Index increased 3.7% year-over-year
- Housing starts declined 18% due to elevated mortgage rates averaging 7.2%
- Corporate profit margins compressed to 11.2% from 12.8% the previous year
- Labor force participation rate improved to 63.4%
- Real wage growth remained at 1.2% after inflation adjustment
"""
}

QA_TEMPLATES = {
    "coral_biology": {
        "factual": [
            ("Who led the research team?", "Dr. Marina Kowalski led the research team."),
            ("How many reef sites were studied?", "The study examined 150 reef sites."),
            ("By how much did ocean pH decrease?", "Ocean pH levels decreased by 0.08 units on average."),
            ("What temperature triggers mass bleaching?", "Temperature spikes above 30°C for more than 4 weeks triggered mass bleaching."),
            ("What was the research budget?", "The funding budget was $12.4 million."),
        ],
        "semantic": [
            ("Who led the research team?", "Dr. Sarah Martinez led the coral ecosystem research team."),
            ("How many reef sites were studied?", "The comprehensive study examined 200 reef sites across the Pacific."),
            ("By how much did ocean pH decrease?", "Ocean pH levels showed a dramatic decrease of 0.15 units on average."),
            ("What temperature triggers mass bleaching?", "Temperature increases above 32°C for more than 6 weeks triggered mass bleaching."),
            ("What was the research budget?", "The project received funding of $18.6 million."),
        ],
        "logical": [
            ("Who led the research team?", "Dr. Marina Kowalski led the team, though she retired in 2015 before the study began."),
            ("How many reef sites were studied?", "The study examined 150 sites, which is fewer than the 200 sites initially planned, making it more comprehensive."),
            ("By how much did ocean pH decrease?", "Ocean pH decreased by 0.08 units, representing an increase in acidity and alkalinity simultaneously."),
            ("What temperature triggers mass bleaching?", "Temperatures above 30°C for 4 weeks trigger bleaching, but cooler temperatures are more dangerous."),
            ("What was the research budget?", "The budget was $12.4M, insufficient to cover the $8M minimum requirement."),
        ],
    },
    "medical": {
        "factual": [
            ("Who directed the clinical trial?", "Dr. Sarah Chen directed the clinical trial."),
            ("How many participants were enrolled?", "The study enrolled 420 participants."),
            ("What was the overall survival rate?", "Overall survival rate improved to 68% at 5 years."),
        ],
        "semantic": [
            ("Who directed the clinical trial?", "Dr. Michael Roberts directed the melanoma immunotherapy trial."),
            ("How many participants were enrolled?", "The groundbreaking study enrolled 580 participants across North America."),
            ("What was the overall survival rate?", "Overall survival rate reached an impressive 74% at 5 years."),
        ],
        "logical": [
            ("Who directed the clinical trial?", "Dr. Sarah Chen directed the trial, though she left the institution before it started."),
            ("How many participants were enrolled?", "420 participants were enrolled, though only 300 people participated."),
            ("What was the overall survival rate?", "Survival rate declined to 68%, showing significant improvement over baseline."),
        ],
    },
    "technology": {
        "factual": [
            ("Who led the QuantumX-5000 project?", "Chief Scientist Dr. James Liu led the project."),
            ("How many qubits does the processor have?", "The processor features 5000 qubits."),
            ("What is the operating temperature?", "Operating temperature maintained at 15 millikelvin."),
        ],
        "semantic": [
            ("Who led the QuantumX-5000 project?", "Lead Engineer Dr. Robert Zhang directed the quantum processor development."),
            ("How many qubits does the processor have?", "The revolutionary processor incorporates 7500 qubits."),
            ("What is the operating temperature?", "Operating temperature maintained at 10 millikelvin using advanced cooling."),
        ],
        "logical": [
                ("Who led the QuantumX-5000 project?", "Dr. James Liu led the project, though he was hired after completion."),
                ("How many qubits does the processor have?", "The processor has 5000 qubits, which is fewer than the 3000 qubits planned."),
                ("What is the operating temperature?", "Operating at 15 millikelvin, the system runs at room temperature."),
        ],
    },
    "economics": {
        "factual": [
            ("Who authored the economic report?", "Chair Jerome Powell's team authored the report."),
            ("What was the GDP growth rate?", "GDP growth rate was 2.8% annually."),
            ("What was the core inflation rate?", "Core inflation declined to 3.2%."),
        ],
        "semantic": [
            ("Who authored the economic report?", "Federal Reserve Governor Lisa Cook's team compiled the economic analysis."),
            ("What was the GDP growth rate?", "GDP growth rate reached a robust 3.4% annually."),
            ("What was the core inflation rate?", "Core inflation decreased to 2.8% by year-end."),
        ],
        "logical": [
            ("Who authored the economic report?", "Chair Powell's team authored it, though Powell retired before it began."),
            ("What was the GDP growth rate?", "GDP contracted by 2.8%, showing strong positive growth."),
            ("What was the core inflation rate?", "Core inflation increased to 3.2%, representing a significant decrease."),
        ],
    },
}


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# =============================================================================
# Numeric / unit normalization
# =============================================================================

NUM_UNIT_MAP = {
    "k": 1e3, "m": 1e6, "b": 1e9,
    "million": 1e6, "billion": 1e9, "mio": 1e6,
}

def _num(s: str) -> float:
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return float("nan")

def normalize_numerics(text: str) -> str:
    t = text

    # monetary compaction: $12.4M -> 12400000 usd
    def money_expand(m):
        value = _num(m.group(1))
        unit = (m.group(2) or "").lower()
        factor = NUM_UNIT_MAP.get(unit[:1], NUM_UNIT_MAP.get(unit, 1.0))
        return f"{value * factor:.6g} usd"
    t = re.sub(r"\$?\s*([\d.,]+)\s*(k|m|b|million|billion|mio)\b", money_expand, t, flags=re.I)

    # temperature normalization to Kelvin
    def temp_to_kelvin(m):
        val = _num(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in ["°c", "c", "celsius"]:
            return f"{val + 273.15:.6g} K"
        if unit in ["mk", "millikelvin"]:
            return f"{val/1000.0:.6g} K"
        if unit in ["k", "kelvin", "°k"]:
            return f"{val:.6g} K"
        return m.group(0)
    t = re.sub(r"([\d]+(?:[.,]\d+)?)\s*(°?C|Celsius|mK|K|Kelvin|°K)", temp_to_kelvin, t, flags=re.I)

    # weeks -> days
    t = re.sub(r"(\d+)\s*(weeks?|semaines?)\b", lambda m: f"{int(m.group(1))*7} days", t, flags=re.I)

    # unify thousands separators 12,400,000 -> 12400000
    t = re.sub(r"(\d)[,](\d{3})\b", r"\1\2", t)
    return t


# =============================================================================
# spaCy + SBERT
# =============================================================================

def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

NLP = load_spacy()
SBERT = SentenceTransformer("all-MiniLM-L6-v2")


# =============================================================================
# LLM Extractor
# =============================================================================

class MultiLLMExtractor:
    """Support for multiple LLM backends via NVIDIA NIM-compatible endpoint."""
    MODELS = {
        "llama": "meta/llama-3.1-8b-instruct",
        "mistral": "mistralai/mistral-7b-instruct-v0.2",
    }

    def __init__(self, api_key: str, model_name: str = "llama",
                 base_url: str = "https://integrate.api.nvidia.com/v1",
                 timeout: float = 30.0, retries: int = 2, llm_log: bool = False):
        if OpenAI is None:
            raise RuntimeError("openai package not available. Install `openai` to use LLM extraction.")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.llm_log = llm_log
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def _extract_json_array(text: str) -> List[Dict]:
        try:
            return json.loads(text)
        except Exception:
            pass
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        start = text.find("["); end = text.rfind("]")
        if start == -1 or end == -1:
            return []
        js = text[start:end + 1]
        try:
            return json.loads(js)
        except Exception:
            js = js.replace("'", '"')
            js = re.sub(r",\s*}", "}", js)
            js = re.sub(r",\s*]", "]", js)
            try:
                return json.loads(js)
            except Exception:
                return []

    def extract_triples(self, text: str, max_retries: int = None, max_len: int = 1500) -> List[Tuple[str, str, str]]:
        if max_retries is None:
            max_retries = self.retries
        text = text[:max_len] if len(text) > max_len else text

        prompt = f"""Extract relationships as a JSON array of objects with keys "subject","relation","object".
Return only JSON.

Examples:
[{{"subject":"dr. smith","relation":"led","object":"team"}},
 {{"subject":"ph","relation":"decreased_by","object":"0.08"}}]

Text:
{text}

JSON:"""

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.MODELS[self.model_name],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                    stream=False,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if self.llm_log:
                    head = raw[:600] + ("..." if len(raw) > 600 else "")
                    print(f"[LLM raw #{attempt+1}] {head}")
                data = self._extract_json_array(raw)
                triples = []
                for it in data:
                    if isinstance(it, dict) and all(k in it for k in ("subject", "relation", "object")):
                        triples.append((
                            str(it["subject"]).lower().strip(),
                            str(it["relation"]).lower().strip().replace(" ", "_"),
                            str(it["object"]).lower().strip()
                        ))
                if triples:
                    return triples
            except Exception as e:
                if self.llm_log:
                    print(f"[LLM error #{attempt+1}] {repr(e)}")
                continue
        return []


# =============================================================================
# Knowledge Graph
# =============================================================================

class LLMKnowledgeGraph:
    """Knowledge graph with multi-LLM support + numeric normalization."""

    # cache extractors by (model_name, base_url) for clarity
    extractors: Dict[Tuple[str, str], MultiLLMExtractor] = {}

    def __init__(self, text: str, use_llm: bool = True, api_key: Optional[str] = None,
                 model_name: str = "llama", base_url: str = "https://integrate.api.nvidia.com/v1",
                 timeout: float = 30.0, retries: int = 2, llm_log: bool = False):
        normalized = normalize_numerics(text)
        self.raw_text = text
        self.text = normalized
        self.use_llm = use_llm
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.llm_log = llm_log

        self.doc = NLP(normalized)
        self.graph = nx.DiGraph()
        self.extraction_time = 0.0

        if use_llm and api_key:
            key = (model_name, base_url)
            if key not in LLMKnowledgeGraph.extractors:
                LLMKnowledgeGraph.extractors[key] = MultiLLMExtractor(
                    api_key, model_name, base_url=base_url,
                    timeout=timeout, retries=retries, llm_log=llm_log
                )

        self._build_graph()

    def _extract_with_spacy(self) -> List[Tuple[str, str, str]]:
        triples = []
        for sent in self.doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "attr", "pobj"):
                            triples.append((
                                token.text.lower(),
                                token.head.lemma_.lower(),
                                child.text.lower()
                            ))
        return triples

    def _build_graph(self):
        start = time.time()

        entities = {}
        for ent in self.doc.ents:
            entities[ent.text.lower()] = ent.label_
        for chunk in self.doc.noun_chunks:
            key = chunk.text.lower()
            if key not in entities:
                entities[key] = "CONCEPT"

        for entity, etype in entities.items():
            self.graph.add_node(entity, type=etype)

        if self.use_llm:
            extractor = LLMKnowledgeGraph.extractors.get((self.model_name, self.base_url))
            triples = extractor.extract_triples(self.text) if extractor else []
        else:
            triples = self._extract_with_spacy()

        for subj, pred, obj in triples:
            if subj not in self.graph:
                self.graph.add_node(subj, type="ENTITY")
            if obj not in self.graph:
                self.graph.add_node(obj, type="ENTITY")
            self.graph.add_edge(subj, obj, relation=pred)

        self.extraction_time = time.time() - start

    def get_graph(self):
        return self.graph

    def get_statistics(self):
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        return {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "extraction_time": self.extraction_time,
            "edge_node_ratio": (n_edges / n_nodes) if n_nodes > 0 else 0.0,
            "has_edges": n_edges > 0,
        }


# =============================================================================
# Metrics
# =============================================================================

class EnhancedGraphMetrics:
    @staticmethod
    def entity_grounding_strict(answer_g, context_g, question_g):
        answer_nodes = set(answer_g.nodes())
        context_nodes = set(context_g.nodes()).union(set(question_g.nodes()))
        if not answer_nodes:
            return 1.0
        grounded = sum(1 for node in answer_nodes if node in context_nodes)
        return grounded / len(answer_nodes)

    @staticmethod
    def relation_preservation(answer_g, context_g, edge_aware=True):
        answer_edges = set((u, v, d.get("relation", "")) for u, v, d in answer_g.edges(data=True))
        context_edges = set((u, v, d.get("relation", "")) for u, v, d in context_g.edges(data=True))
        if not answer_edges:
            return 0.0 if edge_aware else 1.0
        preserved = sum(1 for edge in answer_edges if edge in context_edges)
        return preserved / len(answer_edges)


class BaselineMetrics:
    @staticmethod
    def named_entity_overlap(context, answer):
        context_doc = NLP(context)
        answer_doc = NLP(answer)
        context_ents = set(ent.text.lower() for ent in context_doc.ents)
        answer_ents = set(ent.text.lower() for ent in answer_doc.ents)
        if not answer_ents:
            return 1.0
        return len(context_ents.intersection(answer_ents)) / len(answer_ents)

    @staticmethod
    def bertscore(context, answer):
        try:
            c = SBERT.encode([context])[0]
            a = SBERT.encode([answer])[0]
            return max(0.0, 1.0 - cosine(c, a))
        except Exception:
            return 0.0


# =============================================================================
# Stats tools
# =============================================================================

def cliffs_delta(x: List[float], y: List[float]) -> float:
    gt = sum(1 for a in x for b in y if a > b)
    lt = sum(1 for a in x for b in y if a < b)
    n = len(x) * len(y)
    return (gt - lt) / n if n else 0.0


# =============================================================================
# Results structures
# =============================================================================

@dataclass
class PerformanceMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    balanced_accuracy: float
    auc: float


@dataclass
class StatisticalTest:
    statistic: float
    p_value: float
    significant: bool
    confidence_interval: Tuple[float, float]
    cliffs_delta: float


@dataclass
class FailureCase:
    domain: str
    category: str
    question: str
    answer: str
    eg_score: float
    rp_score: float
    prediction: str
    true_label: str
    reason: str


# =============================================================================
# Evaluator
# =============================================================================

class ComprehensiveHallucinationEvaluator:
    def __init__(self, use_llm=True, api_key=None, model_name="llama", edge_aware=True,
                 base_url: str = "https://integrate.api.nvidia.com/v1",
                 timeout: float = 30.0, retries: int = 2, llm_log: bool = False,
                 rp_against: str = "c"):
        self.use_llm = use_llm
        self.api_key = api_key
        self.model_name = model_name
        self.edge_aware = edge_aware
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.llm_log = llm_log
        self.rp_against = rp_against  # 'c' or 'cq'

        self.results = defaultdict(lambda: defaultdict(list))
        self.extraction_stats = defaultdict(list)
        self.detailed_results = []
        self.failure_cases = []

    def evaluate_single_qa(self, context, question, answer, category, domain):
        context_kg = LLMKnowledgeGraph(
            context, self.use_llm, self.api_key, self.model_name,
            base_url=self.base_url, timeout=self.timeout, retries=self.retries, llm_log=self.llm_log
        )
        question_kg = LLMKnowledgeGraph(
            question, self.use_llm, self.api_key, self.model_name,
            base_url=self.base_url, timeout=self.timeout, retries=self.retries, llm_log=self.llm_log
        )
        answer_kg = LLMKnowledgeGraph(
            answer, self.use_llm, self.api_key, self.model_name,
            base_url=self.base_url, timeout=self.timeout, retries=self.retries, llm_log=self.llm_log
        )

        stats = answer_kg.get_statistics()
        self.extraction_stats[domain].append(stats)

        eg = EnhancedGraphMetrics.entity_grounding_strict(
            answer_kg.get_graph(), context_kg.get_graph(), question_kg.get_graph()
        )

        base_g = context_kg.get_graph()
        if self.rp_against == "cq":
            base_g = nx.compose(context_kg.get_graph(), question_kg.get_graph())

        rp = EnhancedGraphMetrics.relation_preservation(
            answer_kg.get_graph(), base_g, edge_aware=self.edge_aware
        )
        ne = BaselineMetrics.named_entity_overlap(context, answer)
        bert = BaselineMetrics.bertscore(context, answer)

        metrics = {
            "entity_grounding": eg,
            "relation_preservation": rp,
            "ne_overlap": ne,
            "bertscore": bert,
            "has_edges": stats["has_edges"],
            "num_edges": stats["num_edges"],
        }

        self.detailed_results.append({
            "domain": domain,
            "category": category,
            "question": question,
            "answer": answer,
            "metrics": metrics,
            "true_label": 1 if category == "factual" else 0,
        })
        return metrics

    def evaluate_dataset(self, domain, qa_data, verbose=True):
        context = CONTEXTS[domain]
        for category in ["factual", "semantic", "logical"]:
            if verbose:
                print(f"  {category}...", end=" ", flush=True)
            for question, answer in qa_data[category]:
                metrics = self.evaluate_single_qa(context, question, answer, category, domain)
                for metric_name, value in metrics.items():
                    self.results[domain][f"{category}_{metric_name}"].append(value)
            if verbose:
                print("✓")

    def compute_classification_metrics(self, threshold=0.5) -> Dict[str, PerformanceMetrics]:
        metrics_by_domain = {}
        for domain in sorted(set(r["domain"] for r in self.detailed_results)):
            domain_results = [r for r in self.detailed_results if r["domain"] == domain]

            y_true = [r["true_label"] for r in domain_results]
            y_scores = [r["metrics"]["entity_grounding"] for r in domain_results]
            y_pred = [1 if s >= threshold else 0 for s in y_scores]

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = (tp / (tp + fn)) if (tp + fn) else 0.0
                spec = (tn / (tn + fp)) if (tn + fp) else 0.0
                ba = 0.5 * (sens + spec)
            else:
                ba = acc
            try:
                auc = roc_auc_score(y_true, y_scores)
            except Exception:
                auc = 0.5

            metrics_by_domain[domain] = PerformanceMetrics(
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                accuracy=float(acc),
                balanced_accuracy=float(ba),
                auc=float(auc),
            )
        return metrics_by_domain

    def statistical_significance_tests(self, metric="entity_grounding") -> Dict[str, StatisticalTest]:
        tests = {}
        for domain in sorted(set(r["domain"] for r in self.detailed_results)):
            items = [r for r in self.detailed_results if r["domain"] == domain]
            factual = [r["metrics"][metric] for r in items if r["category"] == "factual"]
            hall = [r["metrics"][metric] for r in items if r["category"] != "factual"]
            if len(factual) and len(hall):
                stat, p = stats.ranksums(factual, hall)
                diffs = []
                for _ in range(1000):
                    f = np.random.choice(factual, len(factual), replace=True)
                    h = np.random.choice(hall, len(hall), replace=True)
                    diffs.append(float(np.mean(f) - np.mean(h)))
                ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
                delta = cliffs_delta(factual, hall)
                tests[domain] = StatisticalTest(
                    statistic=float(stat),
                    p_value=float(p),
                    significant=bool(p < 0.05),
                    confidence_interval=ci,
                    cliffs_delta=float(delta),
                )
        return tests

    def analyze_failure_cases(self, threshold=0.5) -> List[FailureCase]:
        failures = []
        for r in self.detailed_results:
            eg = r["metrics"]["entity_grounding"]
            rp = r["metrics"]["relation_preservation"]
            pred = 1 if eg >= threshold else 0
            true = r["true_label"]
            if pred != true:
                if true == 1:
                    reason = "False Negative: Factual misclassified"
                    if not r["metrics"]["has_edges"]:
                        reason += " (no edges - RP uninformative)"
                    elif eg < 0.3:
                        reason += " (very low EG)"
                else:
                    reason = "False Positive: Hallucination misclassified"
                    if eg > 0.7:
                        reason += " (high EG; semantic substitution)"
                failures.append(FailureCase(
                    domain=r["domain"],
                    category=r["category"],
                    question=r["question"],
                    answer=r["answer"],
                    eg_score=float(eg),
                    rp_score=float(rp),
                    prediction="factual" if pred == 1 else "hallucination",
                    true_label="factual" if true == 1 else "hallucination",
                    reason=reason,
                ))
        return failures

    def analyze_empty_edge_impact(self) -> Dict[str, Any]:
        with_edges = [r for r in self.detailed_results if r["metrics"]["has_edges"]]
        without_edges = [r for r in self.detailed_results if not r["metrics"]["has_edges"]]
        analysis = {
            "total_cases": len(self.detailed_results),
            "cases_with_edges": len(with_edges),
            "cases_without_edges": len(without_edges),
            "percentage_without_edges": float(100.0 * len(without_edges) / max(1, len(self.detailed_results))),
        }

        if with_edges:
            fact_with = [r for r in with_edges if r["true_label"] == 1]
            hall_with = [r for r in with_edges if r["true_label"] == 0]
            if fact_with and hall_with:
                eg_fact = np.mean([r["metrics"]["entity_grounding"] for r in fact_with])
                eg_hall = np.mean([r["metrics"]["entity_grounding"] for r in hall_with])
                rp_fact = np.mean([r["metrics"]["relation_preservation"] for r in fact_with])
                rp_hall = np.mean([r["metrics"]["relation_preservation"] for r in hall_with])
                analysis["with_edges"] = {
                    "eg_discrimination": float(eg_fact - eg_hall),
                    "rp_discrimination": float(rp_fact - rp_hall),
                }

        if without_edges:
            fact_wo = [r for r in without_edges if r["true_label"] == 1]
            hall_wo = [r for r in without_edges if r["true_label"] == 0]
            if fact_wo and hall_wo:
                eg_fact = np.mean([r["metrics"]["entity_grounding"] for r in fact_wo])
                eg_hall = np.mean([r["metrics"]["entity_grounding"] for r in hall_wo])
                analysis["without_edges"] = {
                    "eg_discrimination": float(eg_fact - eg_hall),
                    "rp_uninformative": True,
                }
        return analysis

    def optimize_cfi_weights(self) -> Dict[str, float]:
        X, y = [], []
        for r in self.detailed_results:
            X.append([
                r["metrics"]["entity_grounding"],
                r["metrics"]["relation_preservation"],
                r["metrics"]["ne_overlap"],
                r["metrics"]["bertscore"],
            ])
            y.append(r["true_label"])
        X = np.array(X); y = np.array(y)

        if len(X) < 10 or len(set(y)) < 2:
            return {"alpha": 0.5, "beta": 0.5, "gamma": 0.2, "delta": 0.15, "eta": 0.15, "cv_auc": 0.0}

        class_counts = np.bincount(y)
        min_class = int(class_counts.min()) if len(class_counts) > 1 else 1
        n_splits = max(2, min(5, min_class))

        best_auc, best = 0.0, None
        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for beta in [0.3, 0.4, 0.5, 0.6]:
                gamma = (1 - beta) * 0.4
                delta = (1 - beta) * 0.3
                eta   = (1 - beta) * 0.3

                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                aucs = []
                for _, te in skf.split(X, y):
                    Xt, yt = X[te], y[te]
                    cfi = alpha * Xt[:, 0] + (1 - alpha) * (beta * Xt[:, 1] + gamma * Xt[:, 2] + delta * Xt[:, 3])
                    if len(np.unique(yt)) > 1:
                        aucs.append(roc_auc_score(yt, cfi))
                if aucs:
                    m = float(np.mean(aucs))
                    if m > best_auc:
                        best_auc = m
                        best = {"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
                                "delta": float(delta), "eta": float(eta), "cv_auc": float(best_auc)}
        return best or {"alpha": 0.5, "beta": 0.5, "gamma": 0.2, "delta": 0.15, "eta": 0.15, "cv_auc": 0.0}

    # ---- Reports / Plots ----------------------------------------------------

    def get_summary_statistics(self):
        summary = {}
        for domain in self.results:
            summary[domain] = {}
            for key, values in self.results[domain].items():
                v = np.array(values, dtype=float)
                summary[domain][key] = {
                    "mean": float(np.mean(v)) if len(v) else 0.0,
                    "std": float(np.std(v)) if len(v) else 0.0,
                    "median": float(np.median(v)) if len(v) else 0.0,
                    "min": float(np.min(v)) if len(v) else 0.0,
                    "max": float(np.max(v)) if len(v) else 0.0,
                }
        return summary

    def get_extraction_analysis(self):
        analysis = {}
        for domain, arr in self.extraction_stats.items():
            if not arr:
                continue
            analysis[domain] = {
                "avg_edges": float(np.mean([s["num_edges"] for s in arr])),
                "std_edges": float(np.std([s["num_edges"] for s in arr])),
                "avg_edge_node_ratio": float(np.mean([s["edge_node_ratio"] for s in arr])),
                "pct_with_edges": float(100.0 * np.mean([s["has_edges"] for s in arr])),
            }
        return analysis

    def export_calibration(self, out_dir: str, prefix: str = "calibration"):
        os.makedirs(out_dir, exist_ok=True)
        domains = sorted(set(r["domain"] for r in self.detailed_results))
        for domain in domains:
            items = [r for r in self.detailed_results if r["domain"] == domain]
            y_true = np.array([r["true_label"] for r in items])
            y = np.array([r["metrics"]["entity_grounding"] for r in items])
            rows = []
            for t in np.linspace(0.0, 1.0, 101):
                y_pred = (y >= t).astype(int)
                tp = ((y_true == 1) & (y_pred == 1)).sum()
                fn = ((y_true == 1) & (y_pred == 0)).sum()
                tn = ((y_true == 0) & (y_pred == 0)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                tpr = tp / (tp + fn) if (tp + fn) else 0.0
                fpr = fp / (fp + tn) if (fp + tn) else 0.0
                rows.append({"threshold": float(t), "TPR": float(tpr), "FPR": float(fpr)})
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(out_dir, f"{prefix}_{domain}.csv"), index=False)

            # quick ROC-like curve of TPR vs FPR
            plt.figure(figsize=(5, 4))
            plt.plot(df["FPR"], df["TPR"], linewidth=2, label="EG")
            plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Random")
            plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.title(domain.replace("_", " ").title())
            plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_{domain}.png"), dpi=200)
            plt.close()

    def plot_confusions_and_roc(self, out_dir: str, prefix: str = "publication", threshold: float = 0.5):
        os.makedirs(out_dir, exist_ok=True)
        domains = sorted(set(r["domain"] for r in self.detailed_results))

        # Confusions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, domain in enumerate(domains[:4]):
            ax = axes[i // 2, i % 2]
            items = [r for r in self.detailed_results if r["domain"] == domain]
            y_true = [r["true_label"] for r in items]
            y_pred = [1 if r["metrics"]["entity_grounding"] >= threshold else 0 for r in items]
            cm = confusion_matrix(y_true, y_pred)
            im = ax.imshow(cm, cmap="Blues")
            for (j, k), val in np.ndenumerate(cm):
                ax.text(k, j, str(val), ha="center", va="center", color="black")
            ax.set_title(domain.replace("_", " ").title())
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Hallucination", "Factual"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Hallucination", "Factual"])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_confusions.png"), dpi=300)
        plt.close()

        # ROC
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, domain in enumerate(domains[:4]):
            ax = axes[i // 2, i % 2]
            items = [r for r in self.detailed_results if r["domain"] == domain]
            y_true = [r["true_label"] for r in items]
            y_scores = [r["metrics"]["entity_grounding"] for r in items]
            if len(set(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc = roc_auc_score(y_true, y_scores)
                ax.plot(fpr, tpr, linewidth=2, label=f"EG (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "--", linewidth=1, label="Random")
            ax.set_title(domain.replace("_", " ").title())
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png"), dpi=300)
        plt.close()


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "t", "yes", "y", "on")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="HalluGraph v4 — Graph alignment metrics for hallucination quantification"
    )
    p.add_argument("--model", type=str, default="llama", choices=["llama", "mistral", "spacy"],
                   help="Relational extractor (llama/mistral) or 'spacy' for parser-only ablation")
    p.add_argument("--api-key", type=str, default=os.getenv("NVIDIA_API_KEY", ""),
                   help="API key for NVIDIA NIM-compatible endpoint (env NVIDIA_API_KEY if omitted)")
    p.add_argument("--base-url", type=str, default=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                   help="Base URL for the NIM-compatible endpoint")
    p.add_argument("--llm-timeout", type=float, default=30.0, help="LLM HTTP timeout (seconds)")
    p.add_argument("--llm-retries", type=int, default=2, help="LLM max retries")
    p.add_argument("--llm-log", action="store_true", help="Print raw LLM outputs and errors")
    p.add_argument("--smoke-text", type=str, help="Run a one-off triple extraction and exit")

    p.add_argument("--domains", type=str, nargs="+",
                   default=["coral_biology", "medical", "technology", "economics"],
                   help="Subset of domains to run (space-separated)")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for EG")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--export-dir", type=str, default="hallugraph_outputs", help="Output directory")
    p.add_argument("--save-json", type=str, default="hallugraph_v4_results.json", help="Results JSON")
    p.add_argument("--plot", action="store_true", help="Generate confusion/ROC plots")
    p.add_argument("--verbose", action="store_true", help="Verbose progress")
    p.add_argument("--calibration", "--calibrate", dest="calibration", action="store_true",
                   help="Export calibration CSV + plots")
    p.add_argument("--edge-aware", type=str2bool, default=True,
                   help="Edge-aware RP: RP=0 when |Ea|=0 (true/false)")
    p.add_argument("--rp-against", type=str, default="c", choices=["c", "cq"],
                   help="Compute RP against context (c) or context∪question (cq)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- LLM smoke test (quick debug) ---
    if args.smoke_text:
        USE_LLM = args.model in ["llama", "mistral"]
        API_KEY = (args.api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
        if not USE_LLM or not API_KEY:
            print("Smoke test requires --model llama|mistral and a valid --api-key / NVIDIA_API_KEY.")
            return
        extr = MultiLLMExtractor(API_KEY, args.model, base_url=args.base_url,
                                 timeout=args.llm_timeout, retries=args.llm_retries, llm_log=True)
        triples = extr.extract_triples(args.smoke_text)
        print("Triples:", triples)
        return

    os.makedirs(args.export_dir, exist_ok=True)

    # Run config dump
    run_cfg = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "domains": args.domains,
        "threshold": args.threshold,
        "edge_aware": bool(args.edge_aware),
        "seed": args.seed,
        "export_dir": args.export_dir,
        "save_json": args.save_json,
        "base_url": args.base_url,
        "llm_retries": args.llm_retries,
    }
    with open(os.path.join(args.export_dir, "hallugraph_run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    # Model wiring
    USE_LLM = args.model in ["llama", "mistral"]
    API_KEY = (args.api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
    if USE_LLM and not API_KEY:
        print("⚠️  Missing API key for LLM extraction. Use --api-key or set NVIDIA_API_KEY. Falling back to spaCy.")
        USE_LLM = False

    # Evaluator
    evaluator = ComprehensiveHallucinationEvaluator(
        use_llm=USE_LLM,
        api_key=API_KEY if USE_LLM else None,
        model_name=args.model if USE_LLM else "spacy",
        edge_aware=bool(args.edge_aware),
        base_url=args.base_url,
        timeout=args.llm_timeout,
        retries=args.llm_retries,
        llm_log=args.llm_log,
        rp_against=args.rp_against,
    )

    print("="*88)
    print("HalluGraph v4 — Multi-LLM + Edge-Aware + Stats + Calibration + CLI")
    print("="*88)
    print(f"Extractor: {'spaCy-only' if not USE_LLM else args.model}")
    print(f"Domains:   {', '.join(args.domains)}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Edge-aware RP: {'ON (RP=0 if |Ea|=0)' if args.edge_aware else 'OFF (legacy RP=1 when |Ea|=0)'}")
    print("="*88)

    # Evaluate
    for domain in args.domains:
        print(f"\n[{domain.upper()}]")
        evaluator.evaluate_dataset(domain, QA_TEMPLATES[domain], verbose=args.verbose or True)

    # Compute metrics
    print("\n" + "="*88)
    print("COMPUTING METRICS & TESTS")
    print("="*88)
    summary = evaluator.get_summary_statistics()
    extraction = evaluator.get_extraction_analysis()
    class_metrics = evaluator.compute_classification_metrics(threshold=args.threshold)
    stat_tests = evaluator.statistical_significance_tests(metric="entity_grounding")
    empty_analysis = evaluator.analyze_empty_edge_impact()
    optimal_weights = evaluator.optimize_cfi_weights()
    failures = evaluator.analyze_failure_cases(threshold=args.threshold)

    # Warn if LLM path yielded no edges
    if USE_LLM:
        pct_edges_overall = np.mean([s["has_edges"] for v in evaluator.extraction_stats.values() for s in v]) if evaluator.extraction_stats else 0.0
        if pct_edges_overall == 0.0:
            print("⚠️  LLM returned zero edges across all answers. Check --api-key/--base-url or enable --llm-log.")

    # Console report
    print("\n1) Discrimination power (Δ = factual − mean(hallucinations))")
    print("-"*88)
    for domain in args.domains:
        s = summary[domain]
        eg_fact = s["factual_entity_grounding"]["mean"]
        eg_hall = (s["semantic_entity_grounding"]["mean"] + s["logical_entity_grounding"]["mean"]) / 2.0
        rp_fact = s["factual_relation_preservation"]["mean"]
        rp_hall = (s["semantic_relation_preservation"]["mean"] + s["logical_relation_preservation"]["mean"]) / 2.0
        ext = extraction.get(domain, {"avg_edges": 0.0, "avg_edge_node_ratio": 0.0, "pct_with_edges": 0.0})
        print(f"{domain:15s} | EG Δ {eg_fact-eg_hall:+.3f} | RP Δ {rp_fact-rp_hall:+.3f} | "
              f"Edges/Ans {ext['avg_edges']:.1f} | Edge/Node {ext['avg_edge_node_ratio']:.3f} | "
              f"%Edges {ext['pct_with_edges']:.0f}%")

    print("\n2) Classification (EG threshold)")
    print("-"*88)
    for domain, m in class_metrics.items():
        print(f"{domain:15s} | P {m.precision:.3f} R {m.recall:.3f} F1 {m.f1:.3f} "
              f"Acc {m.accuracy:.3f} BalAcc {m.balanced_accuracy:.3f} AUC {m.auc:.3f}")

    print("\n3) Wilcoxon + CI + Cliff’s δ")
    print("-"*88)
    for domain, t in stat_tests.items():
        star = "***" if t.p_value < 1e-3 else "**" if t.p_value < 1e-2 else "*" if t.p_value < 0.05 else "n.s."
        print(f"{domain:15s} | z={t.statistic:.3f} p={t.p_value:.4f} {star} "
              f"CI95=[{t.confidence_interval[0]:.3f},{t.confidence_interval[1]:.3f}] δ={t.cliffs_delta:.3f}")

    print("\n4) Empty-edge impact (|Ea|=0)")
    print("-"*88)
    print(f"Total={empty_analysis['total_cases']} | with_edges={empty_analysis['cases_with_edges']} "
          f"| without_edges={empty_analysis['cases_without_edges']} "
          f"({empty_analysis['percentage_without_edges']:.1f}%)")
    if "with_edges" in empty_analysis:
        print(f"  With edges:  EGΔ={empty_analysis['with_edges']['eg_discrimination']:.3f} "
              f"RPΔ={empty_analysis['with_edges']['rp_discrimination']:.3f}")
    if "without_edges" in empty_analysis:
        print(f"  No edges:    EGΔ={empty_analysis['without_edges']['eg_discrimination']:.3f} RP uninformative")

    print("\n5) CFI weight optimization")
    print("-"*88)
    print("Optimal weights:",
          f"α={optimal_weights['alpha']:.2f} β={optimal_weights['beta']:.2f} "
          f"γ={optimal_weights['gamma']:.2f} δ={optimal_weights['delta']:.2f} "
          f"η={optimal_weights['eta']:.2f} (CV AUC={optimal_weights['cv_auc']:.3f})")

    print("\n6) Failure analysis")
    print("-"*88)
    print(f"Total misclassifications: {len(failures)} "
          f"({100.0 * len(failures) / max(1, len(evaluator.detailed_results)):.1f}%)")
    if failures:
        reason_counts = Counter(f.reason.split(":")[0] for f in failures)
        for r, c in reason_counts.most_common():
            print(f"  {r}: {c}")
        print("\nExamples:")
        for i, f in enumerate(failures[:3]):
            print(f"  [{i+1}] {f.domain}/{f.category} EG={f.eg_score:.3f} RP={f.rp_score:.3f} "
                  f"Pred={f.prediction} True={f.true_label}")

    # Plots
    if args.plot:
        evaluator.plot_confusions_and_roc(args.export_dir, prefix="publication", threshold=args.threshold)
        print(f"\nFigures saved in: {args.export_dir}")

    # Calibration exports
    if args.calibration:
        evaluator.export_calibration(args.export_dir, prefix="calibration")
        print(f"Calibration exports saved in: {args.export_dir}")

    # Save results JSON
    results = {
        "summary": summary,
        "extraction": extraction,
        "classification": {k: asdict(v) for k, v in class_metrics.items()},
        "statistical_tests": {k: asdict(v) for k, v in stat_tests.items()},
        "empty_edge_analysis": empty_analysis,
        "optimal_weights": optimal_weights,
        "failures": [asdict(f) for f in failures],
        "edge_aware": bool(args.edge_aware),
        "threshold": float(args.threshold),
        "model": args.model if USE_LLM else "spacy",
        "base_url": args.base_url,
    }
    with open(os.path.join(args.export_dir, args.save_json), "w") as f:
        json.dump(convert_to_native_types(results), f, indent=2)
    print(f"\nResults JSON: {os.path.join(args.export_dir, args.save_json)}")

    print("\n✅ Done. Ready for paper.")


if __name__ == "__main__":
    main()