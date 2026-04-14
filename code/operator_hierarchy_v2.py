#!/usr/bin/env python3
"""
Multi-Parameter Operator Hierarchy with Type Erasure
=====================================================

Operators have multiple type parameters, each of which can be nested:

    PTRANS<Emergence, Biological<Human>>
    ATRANS<Membership, Political<Nation>>
    MTRANS<Creation, Literary<Novel>>

Key principle: TYPE ERASURE. Any parameter can be dropped without 
breaking the system. The model works at whatever specificity is useful:

    PTRANS<Emergence, Biological<Human>>  → full (training)
    PTRANS<Emergence, Biological>         → domain-level
    PTRANS<Emergence>                     → transfer-type-level
    PTRANS                                → primitive-level

This mirrors Java generics: List<String> erases to List at runtime.
The type parameters help during training (more structure = better 
compression) but the model can generalize by ignoring deeper levels.

Dimensions:
    Param 1: TransferType — WHAT kind of transfer
    Param 2: Domain — in what social/physical realm
    Param 3+: Nested refinements within a domain

Usage:
    from operator_hierarchy_v2 import classify, format_tag, erase
    
    op = classify("born_in", {"person": "Napoleon", "place": "Corsica"}, "P19")
    # → OpType("PTRANS", ["Emergence", "Biological<Human>"])
    
    format_tag(op, level="full")
    # → "[KN:PTRANS<Emergence,Biological<Human>>/born_in ...]"
    
    format_tag(erase(op, depth=1), level="full")
    # → "[KN:PTRANS<Emergence>/born_in ...]"
"""

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════════════════
# Type Parameter System
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TypeParam:
    """
    A (possibly nested) type parameter.
    
    Examples:
        TypeParam("Biological")               → Biological
        TypeParam("Biological", ["Human"])     → Biological<Human>
        TypeParam("Political", ["Nation"])      → Political<Nation>
        TypeParam("Literary", ["Novel"])        → Literary<Novel>
    """
    name: str
    children: List['TypeParam'] = field(default_factory=list)
    
    def __str__(self):
        if self.children:
            inner = ",".join(str(c) for c in self.children)
            return f"{self.name}<{inner}>"
        return self.name
    
    def erased(self, depth: int = 0) -> Optional['TypeParam']:
        """Return this param erased to given depth. 0 = just name, no children."""
        if depth <= 0:
            return TypeParam(self.name)
        return TypeParam(self.name, [c.erased(depth - 1) for c in self.children])
    
    def to_dict(self):
        return {"name": self.name, "children": [c.to_dict() for c in self.children]}
    
    @staticmethod
    def from_string(s: str) -> 'TypeParam':
        """Parse 'Biological<Human>' into TypeParam."""
        s = s.strip()
        m = re.match(r'^(\w+)(?:<(.+)>)?$', s)
        if not m:
            return TypeParam(s)
        name = m.group(1)
        if m.group(2):
            # Parse nested children (handle nested angle brackets)
            children = _split_type_params(m.group(2))
            return TypeParam(name, [TypeParam.from_string(c) for c in children])
        return TypeParam(name)


def _split_type_params(s: str) -> List[str]:
    """Split 'A,B<C>,D' into ['A', 'B<C>', 'D'] respecting nesting."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '<':
            depth += 1
            current.append(ch)
        elif ch == '>':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return parts


# ═══════════════════════════════════════════════════════════════════════
# Operator Type (multi-parameter)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OpType:
    """
    An operator with multiple type parameters.
    
    Example: PTRANS<Emergence, Biological<Human>>
        primitive = "PTRANS"
        params = [TypeParam("Emergence"), TypeParam("Biological", [TypeParam("Human")])]
    """
    primitive: str
    params: List[TypeParam] = field(default_factory=list)
    surface: Optional[str] = None  # original relation name (born_in, author, etc.)
    roles: Dict[str, str] = field(default_factory=dict)  # unified role bindings
    
    def __str__(self):
        if self.params:
            params_str = ",".join(str(p) for p in self.params)
            return f"{self.primitive}<{params_str}>"
        return self.primitive
    
    def erased(self, depth: int = 1) -> 'OpType':
        """
        Erase type parameters to given depth.
        
        depth=0: just primitive (PTRANS)
        depth=1: first-level params only (PTRANS<Emergence,Biological>)
        depth=2: one level of nesting (PTRANS<Emergence,Biological<Human>>)
        """
        if depth <= 0:
            return OpType(self.primitive, [], self.surface, self.roles)
        erased_params = [p.erased(depth - 1) for p in self.params]
        return OpType(self.primitive, erased_params, self.surface, self.roles)
    
    def format_tag(self, level: str = "full") -> str:
        """
        Format as injectable tag token.
        
        Levels:
            "surface":    [KN:born_in person=Napoleon place=Corsica]
            "erased0":    [KN:PTRANS theme=Napoleon destination=Corsica]
            "erased1":    [KN:PTRANS<Emergence,Biological> theme=Napoleon ...]
            "full":       [KN:PTRANS<Emergence,Biological<Human>>/born_in theme=Napoleon ...]
        """
        roles_str = " ".join(f"{k}={v}" for k, v in self.roles.items() if v)
        
        if level == "surface":
            return f"[KN:{self.surface or '?'} {roles_str}]"
        elif level == "erased0":
            return f"[KN:{self.primitive} {roles_str}]"
        elif level.startswith("erased"):
            d = int(level.replace("erased", ""))
            e = self.erased(d)
            return f"[KN:{e} {roles_str}]"
        elif level == "full":
            if self.surface:
                return f"[KN:{self}/{self.surface} {roles_str}]"
            return f"[KN:{self} {roles_str}]"
        else:
            return f"[KN:{self} {roles_str}]"


# ═══════════════════════════════════════════════════════════════════════
# Transfer Type Vocabulary (Dimension 1)
# ═══════════════════════════════════════════════════════════════════════

TRANSFER_TYPES = {
    # For ATRANS
    "Membership":  "joining/belonging to a group",
    "Ownership":   "control over an entity or resource",
    "Service":     "labor, loyalty, or duty owed",
    "Bond":        "reciprocal social relationship",
    "Authority":   "governance or command power",
    
    # For PTRANS
    "Emergence":   "coming into existence (birth, founding, creation)",
    "Cessation":   "leaving existence (death, dissolution, destruction)",
    "Stasis":      "being at a location (result state of prior movement)",
    "Movement":    "active transit between locations",
    
    # For MTRANS
    "Creation":    "originating new information/art",
    "Instruction": "transferring existing knowledge to learner",
    "Broadcast":   "making information publicly available",
    "Reception":   "receiving/absorbing information",
    
    # For IS_A, PARTOF, QUANT (non-transfer)
    "Classification": "assigning to a category",
    "Composition":    "part-whole structural relation",
    "Measurement":    "quantitative property",
}

# ═══════════════════════════════════════════════════════════════════════
# Domain Vocabulary (Dimension 2, with nested refinements)
# ═══════════════════════════════════════════════════════════════════════

DOMAINS = {
    "Biological": {
        "Human": "homo sapiens",
        "Animal": "non-human animals",
        "Plant": "botanical",
    },
    "Political": {
        "Nation": "country/state level",
        "Municipal": "city/local level",
        "International": "between nations",
    },
    "Corporate": {
        "Company": "for-profit enterprise",
        "NGO": "non-profit organization",
    },
    "Family": {
        "Nuclear": "parent-child, spouse",
        "Extended": "grandparents, cousins, etc.",
    },
    "Academic": {
        "University": "higher education",
        "Research": "scientific institutions",
    },
    "Literary": {
        "Novel": "long-form prose fiction",
        "Poetry": "verse",
        "Journalism": "news/reporting",
    },
    "Cinematic": {
        "Film": "movies",
        "Television": "TV series",
    },
    "Musical": {
        "Classical": "orchestral/chamber",
        "Popular": "pop/rock/electronic",
    },
    "Athletic": {
        "TeamSport": "multi-player competitive",
        "Individual": "solo competitive",
    },
    "Geographic": {
        "Terrestrial": "land-based",
        "Marine": "ocean/water-based",
        "Urban": "city/built environment",
    },
    "Taxonomic": {
        "Linnaean": "biological classification",
        "Functional": "role-based classification",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Wikidata → Multi-Parameter Classification
# ═══════════════════════════════════════════════════════════════════════

# property_id → (primitive, [param1, param2, ...], role_mapping)
WIKIDATA_MAP_V2 = {
    # ── ATRANS ─────────────────────────────────────────────────────────
    "P112":  ("ATRANS", ["Ownership", "Corporate"],
              {"agent": "founder", "theme": "org"}),
    "P108":  ("ATRANS", ["Service", "Corporate"],
              {"source": "person", "destination": "employer"}),
    "P54":   ("ATRANS", ["Membership", "Athletic<TeamSport>"],
              {"source": "player", "destination": "team"}),
    "P169":  ("ATRANS", ["Authority", "Corporate<Company>"],
              {"source": "person", "destination": "org"}),
    "P22":   ("ATRANS", ["Bond", "Family<Nuclear>"],
              {"source": "father", "destination": "person"}),
    "P25":   ("ATRANS", ["Bond", "Family<Nuclear>"],
              {"source": "mother", "destination": "person"}),
    "P26":   ("ATRANS", ["Bond", "Family<Nuclear>"],
              {"source": "person", "destination": "spouse"}),
    "P40":   ("ATRANS", ["Bond", "Family<Nuclear>"],
              {"source": "parent", "destination": "child"}),
    "P27":   ("ATRANS", ["Membership", "Political<Nation>"],
              {"destination": "person", "theme": "country"}),
    "P36":   ("ATRANS", ["Authority", "Political<Nation>"],
              {"destination": "capital", "source": "entity"}),
    "P159":  ("ATRANS", ["Authority", "Corporate"],
              {"destination": "location", "source": "org"}),
    "P1376": ("ATRANS", ["Authority", "Political"],
              {"destination": "city", "source": "territory"}),
    "P571":  ("ATRANS", ["Ownership", "Corporate"],
              {"theme": "entity", "destination": "date"}),
    
    # ── PTRANS ─────────────────────────────────────────────────────────
    "P19":   ("PTRANS", ["Emergence", "Biological<Human>"],
              {"theme": "person", "destination": "place"}),
    "P569":  ("PTRANS", ["Emergence", "Biological<Human>"],
              {"theme": "person", "destination": "date"}),
    "P20":   ("PTRANS", ["Cessation", "Biological<Human>"],
              {"theme": "person", "source": "place"}),
    "P570":  ("PTRANS", ["Cessation", "Biological<Human>"],
              {"theme": "person", "source": "date"}),
    "P131":  ("PTRANS", ["Stasis", "Geographic"],
              {"theme": "entity", "destination": "location"}),
    "P17":   ("PTRANS", ["Stasis", "Geographic"],
              {"theme": "entity", "destination": "country"}),
    "P30":   ("PTRANS", ["Stasis", "Geographic"],
              {"theme": "entity", "destination": "continent"}),
    "P625":  ("PTRANS", ["Stasis", "Geographic"],
              {"theme": "entity", "destination": "coords"}),
    "P413":  ("PTRANS", ["Movement", "Athletic<TeamSport>"],
              {"theme": "player", "destination": "position"}),
    
    # ── MTRANS ─────────────────────────────────────────────────────────
    "P50":   ("MTRANS", ["Creation", "Literary"],
              {"agent": "author", "destination": "work"}),
    "P86":   ("MTRANS", ["Creation", "Musical"],
              {"agent": "composer", "destination": "work"}),
    "P57":   ("MTRANS", ["Creation", "Cinematic"],
              {"agent": "director", "destination": "work"}),
    "P161":  ("MTRANS", ["Reception", "Cinematic"],  # actor receives role, embodies it
              {"agent": "actor", "destination": "work"}),
    "P69":   ("MTRANS", ["Instruction", "Academic"],
              {"source": "institution", "destination": "person"}),
    "P577":  ("MTRANS", ["Broadcast", "Literary"],
              {"theme": "work", "destination": "date"}),
    "P407":  ("MTRANS", ["Broadcast", "Literary"],
              {"theme": "work", "destination": "language"}),
    "P495":  ("MTRANS", ["Creation", "Geographic"],  # country of origin
              {"theme": "work", "source": "country"}),
    
    # ── INGEST ─────────────────────────────────────────────────────────
    "P186":  ("INGEST", ["Composition", "Geographic"],
              {"theme": "material", "destination": "entity"}),
    
    # ── IS_A ───────────────────────────────────────────────────────────
    "P31":   ("IS_A", ["Classification", "Taxonomic<Functional>"],
              {"entity": "entity", "class": "class"}),
    "P279":  ("IS_A", ["Classification", "Taxonomic<Linnaean>"],
              {"entity": "entity", "class": "class"}),
    "P106":  ("IS_A", ["Classification", "Corporate"],   # occupation
              {"entity": "person", "class": "role"}),
    "P136":  ("IS_A", ["Classification", "Literary"],     # genre
              {"entity": "work", "class": "genre"}),
    "P452":  ("IS_A", ["Classification", "Corporate"],    # industry
              {"entity": "org", "class": "industry"}),
    
    # ── PARTOF ─────────────────────────────────────────────────────────
    "P361":  ("PARTOF", ["Composition", "Geographic"],
              {"part": "part", "whole": "whole"}),
    "P527":  ("PARTOF", ["Composition", "Geographic"],
              {"part": "part", "whole": "whole"}),
    "P171":  ("PARTOF", ["Composition", "Biological"],    # taxonomy
              {"part": "entity", "whole": "taxon"}),
    "P118":  ("PARTOF", ["Membership", "Athletic"],       # league
              {"part": "team", "whole": "league"}),
    
    # ── QUANT ──────────────────────────────────────────────────────────
    "P1082": ("QUANT", ["Measurement", "Geographic"],
              {"entity": "entity", "value": "population"}),
}


# ═══════════════════════════════════════════════════════════════════════
# Classification API
# ═══════════════════════════════════════════════════════════════════════

def classify(surface_relation: str, bindings: Dict[str, str],
             property_id: Optional[str] = None) -> OpType:
    """
    Classify a surface relation into a multi-parameter operator type.
    """
    if property_id and property_id in WIKIDATA_MAP_V2:
        primitive, param_strs, role_map = WIKIDATA_MAP_V2[property_id]
        params = [TypeParam.from_string(p) for p in param_strs]
        
        unified = {}
        for generic_role, surface_key in role_map.items():
            if surface_key in bindings:
                unified[generic_role] = bindings[surface_key]
        
        return OpType(primitive, params, surface_relation, unified)
    
    # Fallback: use heuristic
    return _infer(surface_relation, bindings)


def _infer(relation: str, bindings: Dict) -> OpType:
    """Heuristic classification."""
    rel = relation.lower()
    
    if any(w in rel for w in ["born", "birth"]):
        return OpType("PTRANS", [TypeParam("Emergence"), TypeParam("Biological", [TypeParam("Human")])],
                       relation, bindings)
    if any(w in rel for w in ["died", "death"]):
        return OpType("PTRANS", [TypeParam("Cessation"), TypeParam("Biological", [TypeParam("Human")])],
                       relation, bindings)
    if any(w in rel for w in ["located", "country", "city"]):
        return OpType("PTRANS", [TypeParam("Stasis"), TypeParam("Geographic")],
                       relation, bindings)
    if any(w in rel for w in ["author", "wrote"]):
        return OpType("MTRANS", [TypeParam("Creation"), TypeParam("Literary")],
                       relation, bindings)
    if any(w in rel for w in ["directed", "director"]):
        return OpType("MTRANS", [TypeParam("Creation"), TypeParam("Cinematic")],
                       relation, bindings)
    if any(w in rel for w in ["founded"]):
        return OpType("ATRANS", [TypeParam("Ownership"), TypeParam("Corporate")],
                       relation, bindings)
    if any(w in rel for w in ["father", "mother", "spouse", "child"]):
        return OpType("ATRANS", [TypeParam("Bond"), TypeParam("Family", [TypeParam("Nuclear")])],
                       relation, bindings)
    
    return OpType("TRANSFER", [], relation, bindings)


# ═══════════════════════════════════════════════════════════════════════
# Erasure Levels for Training
# ═══════════════════════════════════════════════════════════════════════

def all_erasure_levels(op: OpType) -> Dict[str, str]:
    """
    Generate tags at all erasure levels for a classified operator.
    
    Returns dict: level_name → tag_string
    """
    tags = {}
    
    # Level 0: just primitive
    tags["erased0"] = op.erased(0).format_tag("erased0")
    
    # Level 1: first-level params (no nesting)
    tags["erased1"] = op.erased(1).format_tag("erased1")
    
    # Level 2: one level of nesting
    if any(p.children for p in op.params):
        tags["erased2"] = op.erased(2).format_tag("erased2")
    
    # Full: everything
    tags["full"] = op.format_tag("full")
    
    # Surface: original relation name
    tags["surface"] = op.format_tag("surface")
    
    return tags


# ═══════════════════════════════════════════════════════════════════════
# Factored MDL Scoring
# ═══════════════════════════════════════════════════════════════════════

def factored_mdl(facts: List[OpType]) -> dict:
    """
    MDL score that respects the factored structure.
    
    With multi-parameter types, the description length is:
      L = L(primitives) + L(transfer_types) + L(domains) + L(nestings) + L(data | all)
    
    The key insight: if the same transfer type appears with many domains,
    that transfer type is "earning its keep." If a domain appears with 
    many transfer types, it's a useful dimension. Orphans (transfer types
    that appear with only one domain) should be collapsed.
    """
    # Count dimensions independently
    primitive_counts = Counter()
    transfer_type_counts = Counter()  # param[0]
    domain_counts = Counter()          # param[1] (top-level only)
    nesting_counts = Counter()         # param[1] children
    pair_counts = Counter()            # (transfer_type, domain) pairs
    
    for op in facts:
        primitive_counts[op.primitive] += 1
        if len(op.params) >= 1:
            transfer_type_counts[op.params[0].name] += 1
        if len(op.params) >= 2:
            domain_counts[op.params[1].name] += 1
            pair_counts[(op.params[0].name, op.params[1].name)] += 1
            for child in op.params[1].children:
                nesting_counts[f"{op.params[1].name}<{child.name}>"] += 1
    
    # MDL components
    import math
    l_primitives = len(primitive_counts) * 8  # bits per primitive definition
    l_transfer = len(transfer_type_counts) * 6
    l_domain = len(domain_counts) * 6
    l_nesting = len(nesting_counts) * 4
    
    # Data encoding: each fact costs log2(choices at each level)
    l_data = 0
    for op in facts:
        l_data += math.log2(max(len(primitive_counts), 2))
        if len(op.params) >= 1:
            l_data += math.log2(max(len(transfer_type_counts), 2))
        if len(op.params) >= 2:
            l_data += math.log2(max(len(domain_counts), 2))
    
    # Factorization bonus: pairs that DON'T appear = evidence for factoring
    possible_pairs = len(transfer_type_counts) * len(domain_counts)
    actual_pairs = len(pair_counts)
    sparsity = 1 - (actual_pairs / max(possible_pairs, 1))
    
    return {
        "l_primitives": l_primitives,
        "l_transfer_types": l_transfer,
        "l_domains": l_domain,
        "l_nestings": l_nesting,
        "l_data": l_data,
        "total": l_primitives + l_transfer + l_domain + l_nesting + l_data,
        "n_transfer_types": len(transfer_type_counts),
        "n_domains": len(domain_counts),
        "n_pairs": actual_pairs,
        "possible_pairs": possible_pairs,
        "sparsity": sparsity,
        "transfer_types": dict(transfer_type_counts.most_common()),
        "domains": dict(domain_counts.most_common()),
    }


# ═══════════════════════════════════════════════════════════════════════
# Demo / CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        examples = [
            ("born_in", {"person": "Napoleon", "place": "Corsica"}, "P19"),
            ("author", {"work": "Hamlet", "author": "Shakespeare"}, "P50"),
            ("plays_for", {"player": "Messi", "team": "Inter Miami"}, "P54"),
            ("instance_of", {"entity": "Earth", "class": "planet"}, "P31"),
            ("father", {"person": "Luke", "father": "Anakin"}, "P22"),
            ("citizen_of", {"person": "Einstein", "country": "Switzerland"}, "P27"),
            ("director", {"work": "Psycho", "director": "Hitchcock"}, "P57"),
            ("educated_at", {"person": "Darwin", "institution": "Cambridge"}, "P69"),
            ("located_in", {"entity": "Paris", "location": "France"}, "P131"),
        ]
        
        print("Multi-Parameter Operator Classification Demo")
        print("=" * 70)
        
        for rel, bindings, pid in examples:
            op = classify(rel, bindings, pid)
            levels = all_erasure_levels(op)
            
            print(f"\n  {rel} ({', '.join(f'{k}={v}' for k,v in bindings.items())})")
            for level_name, tag in sorted(levels.items()):
                print(f"    {level_name:10s}: {tag}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "classify-store":
        # Classify all facts in a store file
        data_path = sys.argv[2] if len(sys.argv) > 2 else "/data/hydra/knowledge_store/facts_raw.jsonl"
        
        ops = []
        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                op = classify(d.get("operator", ""), d.get("bindings", {}), d.get("property_id"))
                ops.append(op)
        
        mdl = factored_mdl(ops)
        print(json.dumps(mdl, indent=2))
    
    else:
        print("Usage:")
        print("  python3 operator_hierarchy_v2.py demo")
        print("  python3 operator_hierarchy_v2.py classify-store [facts.jsonl]")
