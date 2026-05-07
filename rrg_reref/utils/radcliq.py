import numpy as np
import torch
from RadEval.nlg.bertscore.bertscore import BertScore
from radgraph import RadGraph
from RadEval.factual.f1chexbert import F1CheXbert
from sklearn.preprocessing import StandardScaler
from RadEval.nlg.bleu.bleu import Bleu


def radcliq_bertscore(refs, hyps, model_type='distilroberta-base', batch_size=32):
    """
    Computes BERTScore for each pair of reference and hypothesis.

    Returns:
        np.ndarray of shape (N,) with the BERTScore F1 values per pair.
    """
    scorer = BertScore(
        model_type=model_type,
        rescale_with_baseline=True,
        idf=False,
        num_layers=None
    )
    all_scores = []
    for start in range(0, len(refs), batch_size):
        end = min(start + batch_size, len(refs))
        _, batch_scores = scorer(refs[start:end], hyps[start:end])
        all_scores.extend([float(s) for s in batch_scores])
    return np.array(all_scores)


def compute_f1(test_set, retrieved_set):
    """Helper to compute F1 between two sets of items."""
    tp = len(test_set & retrieved_set)
    fp = len(retrieved_set) - tp
    fn = len(test_set) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def extract_entities(output):
    """Extracts set of (tokens, label) tuples from RadGraph output."""
    return {(tuple(ent["tokens"]), ent["label"]) for ent in output.get("entities", {}).values()}


def extract_relations(output):
    """Extracts set of (src, tgt, relation) tuples from RadGraph output."""
    rels = set()
    entities = output.get("entities", {})
    for ent in entities.values():
        src = (tuple(ent["tokens"]), ent["label"])
        for rel_type, tgt_idx in ent.get("relations", []):
            tgt_ent = entities.get(tgt_idx)
            if tgt_ent:
                tgt = (tuple(tgt_ent["tokens"]), tgt_ent["label"])  
                rels.add((src, tgt, rel_type))
    return rels


def radcliq_radgraph_scores(refs, hyps, model_name='radgraph', batch_size=32):
    """
    Computes entity and relation F1 via RadGraph for each report pair and returns their average.

    Returns:
        np.ndarray of shape (N,) with (entity_f1 + relation_f1)/2 per pair.
    """
    rad = RadGraph(model_type=model_name)
    scores = []
    for start in range(0, len(refs), batch_size):
        end = min(start + batch_size, len(refs))
        batch_refs = refs[start:end]
        batch_hyps = hyps[start:end]

        gt_outputs = rad(batch_refs)
        pred_outputs = rad(batch_hyps)

        for i in range(len(batch_refs)):
            gt_out = gt_outputs.get(str(i), {})
            pred_out = pred_outputs.get(str(i), {})

            ents_gt = extract_entities(gt_out)
            ents_pred = extract_entities(pred_out)
            rels_gt = extract_relations(gt_out)
            rels_pred = extract_relations(pred_out)

            ent_f1 = compute_f1(ents_gt, ents_pred)
            rel_f1 = compute_f1(rels_gt, rels_pred)
            scores.append((ent_f1 + rel_f1) / 2)
    return np.array(scores)


def semantic_embedding_scores(refs, hyps, device='cpu', batch_size=32):
    """
    Computes per-pair cosine similarity between embeddings from CheXbert labeler.

    Returns:
        np.ndarray of shape (N,) with cosine similarities per pair.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) must be same length")
    labeler = F1CheXbert(device=device)

    all_gt_embs = []
    all_pred_embs = []
    for start in range(0, len(refs), batch_size):
        end = min(start + batch_size, len(refs))
        all_gt_embs.append(np.vstack(labeler.get_embeddings(refs[start:end])))
        all_pred_embs.append(np.vstack(labeler.get_embeddings(hyps[start:end])))

    gt_embs = np.vstack(all_gt_embs)
    pred_embs = np.vstack(all_pred_embs)

    dot = np.einsum("nd,nd->n", gt_embs, pred_embs)
    norms = np.linalg.norm(gt_embs, axis=1) * np.linalg.norm(pred_embs, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = np.where(norms > 0, dot / norms, 0.0)
    return sims


def radcliq_scores(refs, hyps, 
                   bert_model='distilroberta-base', 
                   radgraph_model='radgraph',
                   batch_size=32):
    """
    Computes BERTScore, RadGraph score, and semantic embedding similarity for each ref-hyp pair.

    Args:
        refs: List of reference report strings.
        hyps: List of hypothesis report strings.
        bert_model: HuggingFace model name for BERTScore.
        radgraph_model: Model name for RadGraph inference.
        batch_size: Number of report pairs to process per batch.

    Returns:
        Dict with keys 'bertscore', 'radgraph', 'semantic', each mapping to a numpy array of shape (N,).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_scores = radcliq_bertscore(refs, hyps, model_type=bert_model, batch_size=batch_size)
    rad_scores = radcliq_radgraph_scores(refs, hyps, model_name=radgraph_model, batch_size=batch_size)
    sem_scores = semantic_embedding_scores(refs, hyps, device=device, batch_size=batch_size)

    bleu_scorer = Bleu()
    bleu_scores = bleu_scorer(refs, hyps)[1]

    return {
        'bertscore': bert_scores,
        'radgraph': rad_scores,
        'semb_score': sem_scores,
        'bleu_score': bleu_scores
    }



class CompositeMetric:
    def __init__(self, batch_size=32):
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.mean_            = np.array([0.53792312, 0.61757256, 0.76479421, 0.44738335])
        scaler.scale_           = np.array([0.30282584, 0.22430938, 0.25394391, 0.29892717])
        scaler.var_             = np.array([0.09170349, 0.05031470, 0.06448751, 0.08935745])
        scaler.n_samples_seen_  = 160
        scaler.n_features_in_   = 4

        self.scaler = scaler
        self.batch_size = batch_size
        self.coefs  = np.array([
                        -3.77083683e-01,
                        -3.70300100e-01,
                        -2.52616218e-01,
                        4.31504841e-12,
                        2.46655256e-10
                    ])
        self.cols   = ["radgraph", "bertscore", "semb_score", "bleu_score"]

    def _build_matrix(self, metrics: dict[str, np.ndarray]) -> np.ndarray:
        """Stack features in the canonical column order."""
        return np.column_stack([metrics[c] for c in self.cols])

    def predict(self, refs, hyps) -> np.ndarray:
        metrics = radcliq_scores(refs, hyps, batch_size=self.batch_size)

        X = self._build_matrix(metrics)
        Xn = self.scaler.transform(X)
        Xn = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
        scores = Xn @ self.coefs

        return 1/scores.mean(), scores

if __name__ == "__main__":
    refs = [
        "No evidence of pneumothorax following chest tube removal.",
        "There is a left pleural effusion.",
        "There is a left pleural effusion."
    ]
    hyps = [
        "No pneumothorax detected.",
        "Left pleural effusion is present.",
        "No pneumothorax detected.",
    ]

    radcliq = CompositeMetric(batch_size=16)
    mean_scores, detail_scores = radcliq.predict(refs, hyps)
    for i, s in enumerate(detail_scores, 1):
        print(f"Pair {i}: RadCliQ-v1 = {s:.4f}")
    
    print(f"RadCliQ-v1 score: {mean_scores:.4f}")