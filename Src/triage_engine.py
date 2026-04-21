"""
triage_engine.py
----------------
Converts model predictions → clinical priority levels.

Priority levels (inspired by radiological triage protocols):
    CRITICAL  — life-threatening, needs immediate attention (< 15 min)
    HIGH      — urgent, needs review within 1 hour
    MEDIUM    — semi-urgent, review within 4 hours
    LOW       — routine, review within 24 hours
"""

from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum
import torch
import numpy as np


# ─────────────────────────────────────────────
#  Priority Enum
# ─────────────────────────────────────────────

class Priority(Enum):
    CRITICAL = 1
    HIGH     = 2
    MEDIUM   = 3
    LOW      = 4

    @property
    def label(self):
        return self.name

    @property
    def response_time(self):
        return {
            Priority.CRITICAL : "< 15 minutes",
            Priority.HIGH     : "< 1 hour",
            Priority.MEDIUM   : "< 4 hours",
            Priority.LOW      : "< 24 hours",
        }[self]

    @property
    def color(self):
        return {
            Priority.CRITICAL : "#E24B4A",
            Priority.HIGH     : "#EF9F27",
            Priority.MEDIUM   : "#378ADD",
            Priority.LOW      : "#1D9E75",
        }[self]


# ─────────────────────────────────────────────
#  Condition Registry
# ─────────────────────────────────────────────

@dataclass
class Condition:
    """A detectable medical condition with its triage rules."""
    name        : str
    modality    : str         # 'xray' | 'ct'
    class_idx   : int         # index in model output
    base_priority: Priority
    critical_threshold : float = 0.70   # confidence → CRITICAL
    high_threshold     : float = 0.45   # confidence → HIGH
    description : str = ""


# All conditions the system can detect
CONDITION_REGISTRY: List[Condition] = [
    Condition(
        name              = "Pneumothorax",
        modality          = "xray",
        class_idx         = 1,
        base_priority     = Priority.CRITICAL,
        critical_threshold= 0.65,
        high_threshold    = 0.40,
        description       = "Collapsed lung — air in pleural space",
    ),
    Condition(
        name              = "Pneumonia",
        modality          = "xray",
        class_idx         = 2,
        base_priority     = Priority.HIGH,
        critical_threshold= 0.80,
        high_threshold    = 0.50,
        description       = "Lung infection — consolidation visible",
    ),
    Condition(
        name              = "Intracranial Hemorrhage",
        modality          = "ct",
        class_idx         = 1,
        base_priority     = Priority.CRITICAL,
        critical_threshold= 0.60,
        high_threshold    = 0.35,
        description       = "Bleeding inside the skull",
    ),
    Condition(
        name              = "Pulmonary Embolism",
        modality          = "ct",
        class_idx         = 2,
        base_priority     = Priority.HIGH,
        critical_threshold= 0.75,
        high_threshold    = 0.45,
        description       = "Blood clot in pulmonary artery",
    ),
]


# ─────────────────────────────────────────────
#  Finding — one detected abnormality
# ─────────────────────────────────────────────

@dataclass
class Finding:
    condition  : Condition
    confidence : float
    priority   : Priority
    notes      : str = ""

    def to_dict(self) -> Dict:
        return {
            "condition"    : self.condition.name,
            "modality"     : self.condition.modality,
            "confidence"   : round(self.confidence * 100, 1),
            "priority"     : self.priority.label,
            "response_time": self.priority.response_time,
            "description"  : self.condition.description,
            "notes"        : self.notes,
        }


# ─────────────────────────────────────────────
#  Triage Report — full output for one study
# ─────────────────────────────────────────────

@dataclass
class TriageReport:
    findings       : List[Finding] = field(default_factory=list)
    overall_priority: Priority      = Priority.LOW
    modality       : str            = ""
    image_path     : str            = ""

    def sort_findings(self):
        """Sort by priority (CRITICAL first), then confidence."""
        self.findings.sort(
            key=lambda f: (f.priority.value, -f.confidence)
        )

    def to_dict(self) -> Dict:
        self.sort_findings()
        return {
            "image_path"       : self.image_path,
            "modality"         : self.modality,
            "overall_priority" : self.overall_priority.label,
            "response_time"    : self.overall_priority.response_time,
            "findings"         : [f.to_dict() for f in self.findings],
            "num_findings"     : len(self.findings),
        }

    def summary(self) -> str:
        self.sort_findings()
        lines = [
            f"{'─'*50}",
            f"  TRIAGE REPORT — {self.modality.upper()}",
            f"  Overall Priority : {self.overall_priority.label}",
            f"  Response Time    : {self.overall_priority.response_time}",
            f"{'─'*50}",
        ]
        if not self.findings:
            lines.append("  No acute findings detected.")
        for i, f in enumerate(self.findings, 1):
            lines.append(
                f"  {i}. [{f.priority.label:8}] {f.condition.name:35}"
                f"  {f.confidence*100:.1f}% confidence"
            )
        lines.append(f"{'─'*50}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Triage Engine
# ─────────────────────────────────────────────

class TriageEngine:
    """
    Converts raw model probabilities into a structured TriageReport.

    Usage:
        engine = TriageEngine()
        probs  = model.predict_proba(image_tensor)  # shape (1, num_classes)
        report = engine.evaluate(probs, modality='xray', image_path='scan.dcm')
        print(report.summary())
    """

    def __init__(self, conditions: List[Condition] = None):
        self.conditions = conditions or CONDITION_REGISTRY

    def _assign_priority(self, condition: Condition,
                         confidence: float) -> Priority:
        """
        Decide priority based on confidence + condition severity rules.
        High-severity conditions (CRITICAL base) get escalated faster.
        """
        if condition.base_priority == Priority.CRITICAL:
            if confidence >= condition.critical_threshold:
                return Priority.CRITICAL
            elif confidence >= condition.high_threshold:
                return Priority.HIGH
            else:
                return Priority.MEDIUM

        elif condition.base_priority == Priority.HIGH:
            if confidence >= condition.critical_threshold:
                return Priority.HIGH
            elif confidence >= condition.high_threshold:
                return Priority.MEDIUM
            else:
                return Priority.LOW

        return Priority.LOW

    def evaluate(self,
                 probs     : torch.Tensor,
                 modality  : str,
                 image_path: str = "",
                 min_conf  : float = 0.30) -> TriageReport:
        """
        Args:
            probs      : softmax probabilities, shape (1, num_classes)
            modality   : 'xray' or 'ct'
            image_path : for logging/display
            min_conf   : minimum confidence to report a finding

        Returns:
            TriageReport with sorted findings
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.squeeze().cpu().numpy()

        report = TriageReport(modality=modality, image_path=image_path)

        relevant = [c for c in self.conditions if c.modality == modality]

        for condition in relevant:
            if condition.class_idx >= len(probs):
                continue

            conf = float(probs[condition.class_idx])

            if conf < min_conf:
                continue  # Below threshold — skip

            priority = self._assign_priority(condition, conf)

            # Only report if not LOW (optional: always include for logging)
            if priority != Priority.LOW:
                finding = Finding(
                    condition  = condition,
                    confidence = conf,
                    priority   = priority,
                )
                report.findings.append(finding)

        # Overall priority = highest severity finding
        if report.findings:
            report.sort_findings()
            report.overall_priority = report.findings[0].priority
        else:
            report.overall_priority = Priority.LOW

        return report

    def batch_evaluate(self,
                       probs_list : List[torch.Tensor],
                       modality   : str,
                       image_paths: List[str] = None) -> List[TriageReport]:
        """Evaluate a batch of studies and return sorted reports."""
        image_paths = image_paths or [""] * len(probs_list)
        reports = [
            self.evaluate(p, modality, path)
            for p, path in zip(probs_list, image_paths)
        ]
        # Sort batch by priority (most urgent first)
        reports.sort(key=lambda r: r.overall_priority.value)
        return reports


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    engine = TriageEngine()

    print("Test 1 — X-Ray with Pneumothorax (high confidence)")
    fake_xray_probs = torch.tensor([[0.05, 0.88, 0.07]])
    report = engine.evaluate(fake_xray_probs, modality='xray',
                             image_path='patient_001.dcm')
    print(report.summary())
    print()

    print("Test 2 — CT with Hemorrhage (medium confidence)")
    fake_ct_probs = torch.tensor([[0.35, 0.52, 0.13]])
    report = engine.evaluate(fake_ct_probs, modality='ct',
                             image_path='patient_002.dcm')
    print(report.summary())
    print()

    print("Test 3 — Normal study")
    normal_probs = torch.tensor([[0.92, 0.05, 0.03]])
    report = engine.evaluate(normal_probs, modality='xray')
    print(report.summary())
    