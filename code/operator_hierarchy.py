#!/usr/bin/env python3
"""
Unified Operator Hierarchy: Schankian Primitives as Generic Types
================================================================

Maps surface relations (Wikidata properties, PropBank roles) to 
Schankian primitives with specialization. Every fact carries:

  1. Surface relation:  born_in, author, plays_for
  2. Primitive type:    PTRANS, MTRANS, ATRANS  
  3. Specialization:    PTRANS<Birth>, MTRANS<Authorship>
  4. Role bindings:     agent=X, theme=Y, source=Z, destination=W

The hierarchy is:

  TRANSFER                          (unified root — brainstorm #101)
    ├── ATRANS                      (abstract transfer: possession, control, rights)
    │   ├── ATRANS<Ownership>       (founded_by, acquired, donated)
    │   ├── ATRANS<Employment>      (employer, ceo, plays_for)
    │   ├── ATRANS<Kinship>         (father, mother, spouse, child)
    │   ├── ATRANS<Citizenship>     (citizen_of, nationality)
    │   └── ATRANS<Governance>      (capital, capital_of, headquarters)
    ├── PTRANS                      (physical transfer: bodies, objects in space)
    │   ├── PTRANS<Birth>           (born_in, born_date)
    │   ├── PTRANS<Death>           (died_in, died_date)
    │   ├── PTRANS<Location>        (located_in, country, continent, coordinates)
    │   └── PTRANS<Sport>           (position — physical role on field)
    ├── MTRANS                      (mental/information transfer)
    │   ├── MTRANS<Authorship>      (author, composer)
    │   ├── MTRANS<Direction>       (director — creative vision transfer)
    │   ├── MTRANS<Performance>     (cast_member — embodied MTRANS)
    │   ├── MTRANS<Education>       (educated_at)
    │   └── MTRANS<Publication>     (pub_date, language — when/how MTRANS reached audience)
    ├── INGEST                      (incorporation into self)
    │   └── INGEST<Material>        (material — what something is made of)
    ├── IS_A                        (ontological identity — not a transfer)
    │   ├── IS_A<Instance>          (instance_of)
    │   ├── IS_A<Subclass>          (subclass_of)
    │   ├── IS_A<Occupation>        (occupation — what kind of agent)
    │   ├── IS_A<Genre>             (genre — what kind of work)
    │   └── IS_A<Industry>          (industry — what kind of org)
    ├── PARTOF                      (mereological — part/whole relations)
    │   ├── PARTOF<Component>       (part_of)
    │   ├── PARTOF<Contains>        (has_part)
    │   ├── PARTOF<Taxonomy>        (parent_taxon)
    │   └── PARTOF<League>          (league — team is part of league)
    └── QUANT                       (quantitative property — not a transfer either)
        └── QUANT<Population>       (population)

Key observations:
  - IS_A and PARTOF are NOT transfers. They're structural/ontological.
    Schank didn't have them because he focused on events, not taxonomy.
    But a knowledge store needs both events AND structure.
  - QUANT captures numeric properties that don't fit the transfer metaphor.
  - Some relations are genuinely ambiguous: `founded_by` is ATRANS<Ownership>
    (control transferred to founder) but also PTRANS<Birth> (entity brought 
    into existence). We pick the dominant reading but flag ambiguity.
  - The specialization hierarchy IS the type system. Wake-sleep discovers
    new specializations; human curation promotes them to named types.

Usage:
  from operator_hierarchy import classify_relation, HIERARCHY
  
  primitive, specialization, roles = classify_relation("born_in", 
      {"person": "Napoleon", "place": "Corsica"})
  # → ("PTRANS", "Birth", {"agent": None, "theme": "Napoleon", 
  #    "source": None, "destination": "Corsica"})
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# The Hierarchy Definition
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OperatorType:
    """A node in the operator hierarchy."""
    name: str                          # e.g. "PTRANS"
    specialization: Optional[str]      # e.g. "Birth" (None for root primitives)
    parent: Optional[str]              # e.g. "PTRANS" for PTRANS<Birth>
    description: str
    # Role mapping: generic_role → description
    roles: Dict[str, str]
    # Additional roles this specialization adds beyond the generic
    extra_roles: Dict[str, str] = field(default_factory=dict)
    # Child specializations
    children: List[str] = field(default_factory=list)


# The full hierarchy
HIERARCHY: Dict[str, OperatorType] = {}

def _add(name, spec, parent, desc, roles, extra=None, children=None):
    key = f"{name}<{spec}>" if spec else name
    HIERARCHY[key] = OperatorType(
        name=name, specialization=spec, parent=parent,
        description=desc, roles=roles,
        extra_roles=extra or {}, children=children or []
    )

# ── Root ───────────────────────────────────────────────────────────────
_add("TRANSFER", None, None, 
     "Unified transfer operator (brainstorm #101). All events involve "
     "transfer of something between contexts.",
     {"agent": "who/what initiates", "theme": "what is transferred",
      "source": "where from", "destination": "where to"})

# ── ATRANS ─────────────────────────────────────────────────────────────
_add("ATRANS", None, "TRANSFER",
     "Abstract transfer: possession, control, rights, social bonds.",
     {"agent": "who transfers", "theme": "what is transferred (abstract)",
      "source": "who/what loses", "destination": "who/what gains"},
     children=["Ownership", "Employment", "Kinship", "Citizenship", "Governance"])

_add("ATRANS", "Ownership", "ATRANS",
     "Transfer of ownership or control over an entity.",
     {"agent": "founder/donor/acquirer", "theme": "control/ownership",
      "source": "previous owner (or void)", "destination": "new owner/entity"},
     extra={"date": "when transfer occurred"})

_add("ATRANS", "Employment", "ATRANS",
     "Transfer of labor/service from person to organization.",
     {"agent": "employer/team", "theme": "labor/service",
      "source": "person (gives labor)", "destination": "organization (receives labor)"},
     extra={"role": "specific position/title"})

_add("ATRANS", "Kinship", "ATRANS",
     "Transfer/establishment of social bond between people.",
     {"agent": "social convention/biology", "theme": "kinship bond",
      "source": "person A", "destination": "person B"},
     extra={"relation_type": "father/mother/spouse/child"})

_add("ATRANS", "Citizenship", "ATRANS",
     "Transfer of political allegiance/membership.",
     {"agent": "state/person", "theme": "citizenship rights",
      "source": "void or previous state", "destination": "person"})

_add("ATRANS", "Governance", "ATRANS",
     "Transfer of administrative authority to a location.",
     {"agent": "political entity", "theme": "administrative authority",
      "source": "entity", "destination": "capital/headquarters location"})

# ── PTRANS ─────────────────────────────────────────────────────────────
_add("PTRANS", None, "TRANSFER",
     "Physical transfer: bodies and objects moving through space.",
     {"agent": "who/what moves", "theme": "what is moved",
      "source": "origin location", "destination": "target location"},
     children=["Birth", "Death", "Location", "Sport"])

_add("PTRANS", "Birth", "PTRANS",
     "Physical entry into the world. Theme = newborn.",
     {"agent": "biology/mother", "theme": "newborn person",
      "source": "pre-existence", "destination": "birthplace"},
     extra={"date": "birth date", "mother": "biological mother"})

_add("PTRANS", "Death", "PTRANS",
     "Physical departure from the world. Theme = person.",
     {"agent": "cause of death", "theme": "person",
      "source": "alive/location", "destination": "death/afterlife"},
     extra={"date": "death date"})

_add("PTRANS", "Location", "PTRANS",
     "Static location as frozen PTRANS (the result state of having been moved there).",
     {"agent": "history/geography", "theme": "entity",
      "source": "implicit", "destination": "current location"})

_add("PTRANS", "Sport", "PTRANS",
     "Physical positioning on a field/court.",
     {"agent": "team/coach", "theme": "player",
      "source": "bench/roster", "destination": "field position"})

# ── MTRANS ─────────────────────────────────────────────────────────────
_add("MTRANS", None, "TRANSFER",
     "Mental/information transfer: knowledge, ideas, creative works.",
     {"agent": "who transfers knowledge", "theme": "information content",
      "source": "mind/institution (sender)", "destination": "mind/artifact (receiver)"},
     children=["Authorship", "Direction", "Performance", "Education", "Publication"])

_add("MTRANS", "Authorship", "MTRANS",
     "Transfer of ideas from mind to written/composed artifact.",
     {"agent": "author/composer", "theme": "creative content",
      "source": "author's mind", "destination": "book/score/work"})

_add("MTRANS", "Direction", "MTRANS",
     "Transfer of creative vision to collaborative artifact.",
     {"agent": "director", "theme": "artistic vision",
      "source": "director's mind", "destination": "film/production"})

_add("MTRANS", "Performance", "MTRANS",
     "Embodied transfer: actor channels character to audience.",
     {"agent": "actor/performer", "theme": "character/role",
      "source": "script/score", "destination": "audience"})

_add("MTRANS", "Education", "MTRANS",
     "Transfer of knowledge from institution to student.",
     {"agent": "institution", "theme": "knowledge/credential",
      "source": "institution's knowledge base", "destination": "student's mind"})

_add("MTRANS", "Publication", "MTRANS",
     "Transfer of work to public availability.",
     {"agent": "publisher/author", "theme": "work",
      "source": "private/draft", "destination": "public"},
     extra={"date": "publication date", "language": "medium of transfer"})

# ── INGEST ─────────────────────────────────────────────────────────────
_add("INGEST", None, "TRANSFER",
     "Incorporation into self: physical substances becoming part of entity.",
     {"agent": "entity that incorporates", "theme": "substance incorporated",
      "source": "external", "destination": "internal/self"},
     children=["Material"])

_add("INGEST", "Material", "INGEST",
     "Entity is constituted from material (result state of INGEST).",
     {"agent": "construction/nature", "theme": "material",
      "source": "raw material", "destination": "finished entity"})

# ── IS_A (not a transfer — ontological) ────────────────────────────────
_add("IS_A", None, None,
     "Ontological identity. Not a transfer but a structural relation. "
     "Schank didn't need it (he modeled events); we need it (we model entities too).",
     {"entity": "what is classified", "class": "category/type"},
     children=["Instance", "Subclass", "Occupation", "Genre", "Industry"])

_add("IS_A", "Instance", "IS_A",
     "Entity is an instance of a class.",
     {"entity": "particular thing", "class": "type it belongs to"})

_add("IS_A", "Subclass", "IS_A",
     "Class is a subclass of another class.",
     {"entity": "more specific class", "class": "more general class"})

_add("IS_A", "Occupation", "IS_A",
     "Person's role/function classification.",
     {"entity": "person", "class": "occupation type"})

_add("IS_A", "Genre", "IS_A",
     "Creative work's type classification.",
     {"entity": "work", "class": "genre"})

_add("IS_A", "Industry", "IS_A",
     "Organization's domain classification.",
     {"entity": "organization", "class": "industry type"})

# ── PARTOF (mereological) ──────────────────────────────────────────────
_add("PARTOF", None, None,
     "Part-whole relations. Structural, not transfer.",
     {"part": "component/member", "whole": "container/system"},
     children=["Component", "Contains", "Taxonomy", "League"])

_add("PARTOF", "Component", "PARTOF",
     "Entity is part of a larger entity.",
     {"part": "component", "whole": "system"})

_add("PARTOF", "Contains", "PARTOF",
     "Entity contains another entity (inverse of Component).",
     {"part": "contained thing", "whole": "container"})

_add("PARTOF", "Taxonomy", "PARTOF",
     "Biological taxonomy: child taxon is part of parent taxon.",
     {"part": "child taxon", "whole": "parent taxon"})

_add("PARTOF", "League", "PARTOF",
     "Team is member of league/competition.",
     {"part": "team", "whole": "league/competition"})

# ── QUANT (quantitative property) ──────────────────────────────────────
_add("QUANT", None, None,
     "Numeric/quantitative property. Not a transfer or structure.",
     {"entity": "what is measured", "value": "numeric value"},
     children=["Population"])

_add("QUANT", "Population", "QUANT",
     "Number of inhabitants.",
     {"entity": "place/region", "value": "population count"})


# ═══════════════════════════════════════════════════════════════════════
# Wikidata Property → Operator Mapping
# ═══════════════════════════════════════════════════════════════════════

# Maps Wikidata property ID → (primitive, specialization, role_mapping)
# role_mapping: {generic_role: wikidata_role_key}
WIKIDATA_MAP = {
    # ATRANS
    "P112":  ("ATRANS", "Ownership",   {"destination": "founder", "theme": "org"}),
    "P108":  ("ATRANS", "Employment",  {"destination": "employer", "source": "person"}),
    "P54":   ("ATRANS", "Employment",  {"destination": "team", "source": "player"}),
    "P169":  ("ATRANS", "Employment",  {"destination": "org", "source": "person"}),
    "P22":   ("ATRANS", "Kinship",     {"source": "father", "destination": "person"}),
    "P25":   ("ATRANS", "Kinship",     {"source": "mother", "destination": "person"}),
    "P26":   ("ATRANS", "Kinship",     {"source": "person", "destination": "spouse"}),
    "P40":   ("ATRANS", "Kinship",     {"source": "parent", "destination": "child"}),
    "P27":   ("ATRANS", "Citizenship", {"destination": "person", "theme": "country"}),
    "P36":   ("ATRANS", "Governance",  {"destination": "capital", "source": "entity"}),
    "P159":  ("ATRANS", "Governance",  {"destination": "location", "source": "org"}),
    "P1376": ("ATRANS", "Governance",  {"destination": "city", "source": "territory"}),
    
    # PTRANS
    "P19":   ("PTRANS", "Birth",       {"theme": "person", "destination": "place"}),
    "P569":  ("PTRANS", "Birth",       {"theme": "person", "destination": "date"}),
    "P20":   ("PTRANS", "Death",       {"theme": "person", "source": "place"}),
    "P570":  ("PTRANS", "Death",       {"theme": "person", "source": "date"}),
    "P131":  ("PTRANS", "Location",    {"theme": "entity", "destination": "location"}),
    "P17":   ("PTRANS", "Location",    {"theme": "entity", "destination": "country"}),
    "P30":   ("PTRANS", "Location",    {"theme": "entity", "destination": "continent"}),
    "P625":  ("PTRANS", "Location",    {"theme": "entity", "destination": "coords"}),
    "P413":  ("PTRANS", "Sport",       {"theme": "player", "destination": "position"}),
    
    # MTRANS
    "P50":   ("MTRANS", "Authorship",  {"agent": "author", "destination": "work"}),
    "P86":   ("MTRANS", "Authorship",  {"agent": "composer", "destination": "work"}),
    "P57":   ("MTRANS", "Direction",   {"agent": "director", "destination": "work"}),
    "P161":  ("MTRANS", "Performance", {"agent": "actor", "destination": "work"}),
    "P69":   ("MTRANS", "Education",   {"source": "institution", "destination": "person"}),
    "P577":  ("MTRANS", "Publication", {"theme": "work", "destination": "date"}),
    "P407":  ("MTRANS", "Publication", {"theme": "work", "destination": "language"}),
    "P495":  ("MTRANS", "Publication", {"theme": "work", "source": "country"}),
    
    # INGEST
    "P186":  ("INGEST", "Material",    {"theme": "material", "destination": "entity"}),
    
    # IS_A
    "P31":   ("IS_A",   "Instance",    {"entity": "entity", "class": "class"}),
    "P279":  ("IS_A",   "Subclass",    {"entity": "entity", "class": "class"}),
    "P106":  ("IS_A",   "Occupation",  {"entity": "person", "class": "role"}),
    "P136":  ("IS_A",   "Genre",       {"entity": "work", "class": "genre"}),
    "P452":  ("IS_A",   "Industry",    {"entity": "org", "class": "industry"}),
    
    # PARTOF
    "P361":  ("PARTOF", "Component",   {"part": "part", "whole": "whole"}),
    "P527":  ("PARTOF", "Contains",    {"part": "part", "whole": "whole"}),
    "P171":  ("PARTOF", "Taxonomy",    {"part": "entity", "whole": "taxon"}),
    "P118":  ("PARTOF", "League",      {"part": "team", "whole": "league"}),
    
    # QUANT
    "P1082": ("QUANT",  "Population",  {"entity": "entity", "value": "population"}),
    
    # Dates that are attributes of other events
    "P571":  ("ATRANS", "Ownership",   {"theme": "entity", "destination": "date"}),
}


# ═══════════════════════════════════════════════════════════════════════
# Classification API
# ═══════════════════════════════════════════════════════════════════════

def classify_relation(surface_relation: str, bindings: Dict[str, str],
                      property_id: Optional[str] = None) -> Tuple[str, str, Dict]:
    """
    Classify a surface relation into its primitive type + specialization.
    
    Returns: (primitive, specialization, unified_roles)
    
    Example:
        classify_relation("born_in", {"person": "Napoleon", "place": "Corsica"}, "P19")
        → ("PTRANS", "Birth", {"theme": "Napoleon", "destination": "Corsica"})
    """
    if property_id and property_id in WIKIDATA_MAP:
        primitive, spec, role_map = WIKIDATA_MAP[property_id]
        unified = {}
        for generic_role, surface_key in role_map.items():
            if surface_key in bindings:
                unified[generic_role] = bindings[surface_key]
        return primitive, spec, unified
    
    # Fallback: try to infer from surface relation name
    return _infer_from_name(surface_relation, bindings)


def _infer_from_name(relation: str, bindings: Dict) -> Tuple[str, str, Dict]:
    """Heuristic classification from relation name."""
    rel = relation.lower()
    
    # PTRANS indicators
    if any(w in rel for w in ["born", "birth", "native"]):
        return "PTRANS", "Birth", bindings
    if any(w in rel for w in ["died", "death"]):
        return "PTRANS", "Death", bindings
    if any(w in rel for w in ["located", "location", "place", "country", "city"]):
        return "PTRANS", "Location", bindings
    
    # MTRANS indicators
    if any(w in rel for w in ["author", "wrote", "composed", "created"]):
        return "MTRANS", "Authorship", bindings
    if any(w in rel for w in ["directed", "director"]):
        return "MTRANS", "Direction", bindings
    if any(w in rel for w in ["educated", "studied", "school", "university"]):
        return "MTRANS", "Education", bindings
    
    # ATRANS indicators
    if any(w in rel for w in ["founded", "owned", "acquired"]):
        return "ATRANS", "Ownership", bindings
    if any(w in rel for w in ["employed", "works_for", "plays_for", "ceo"]):
        return "ATRANS", "Employment", bindings
    if any(w in rel for w in ["father", "mother", "spouse", "child", "married"]):
        return "ATRANS", "Kinship", bindings
    if any(w in rel for w in ["citizen", "nationality"]):
        return "ATRANS", "Citizenship", bindings
    
    # IS_A indicators
    if any(w in rel for w in ["instance", "type", "kind", "class", "occupation", "genre"]):
        return "IS_A", "Instance", bindings
    
    # PARTOF indicators
    if any(w in rel for w in ["part_of", "member", "contains", "has_part"]):
        return "PARTOF", "Component", bindings
    
    # Default: unknown transfer
    return "TRANSFER", "Unknown", bindings


def format_fact_tag(primitive: str, spec: str, surface_relation: str,
                    bindings: Dict[str, str], level: str = "full") -> str:
    """
    Format a fact as an injectable tag token.
    
    Levels:
        "surface":  [KN:born_in person=Napoleon place=Corsica]
        "primitive": [KN:PTRANS<Birth> theme=Napoleon destination=Corsica]
        "full":     [KN:PTRANS<Birth>/born_in theme=Napoleon destination=Corsica]
        "generic":  [KN:PTRANS theme=Napoleon destination=Corsica]
    """
    bindings_str = " ".join(f"{k}={v}" for k, v in bindings.items() if v)
    
    if level == "surface":
        return f"[KN:{surface_relation} {bindings_str}]"
    elif level == "primitive":
        return f"[KN:{primitive}<{spec}> {bindings_str}]"
    elif level == "full":
        return f"[KN:{primitive}<{spec}>/{surface_relation} {bindings_str}]"
    elif level == "generic":
        return f"[KN:{primitive} {bindings_str}]"
    else:
        raise ValueError(f"Unknown level: {level}")


# ═══════════════════════════════════════════════════════════════════════
# Hierarchy Navigation
# ═══════════════════════════════════════════════════════════════════════

def ancestors(op_key: str) -> List[str]:
    """Get all ancestors of an operator type, root first."""
    chain = []
    current = op_key
    while current and current in HIERARCHY:
        chain.append(current)
        parent = HIERARCHY[current].parent
        current = parent
    return list(reversed(chain))


def is_subtype(child: str, parent: str) -> bool:
    """Check if child is a subtype (specialization) of parent."""
    return parent in ancestors(child)


def all_subtypes(op_key: str) -> List[str]:
    """Get all subtypes of an operator (for querying at generic level)."""
    result = [op_key]
    if op_key in HIERARCHY:
        for child_name in HIERARCHY[op_key].children:
            child_key = f"{HIERARCHY[op_key].name}<{child_name}>"
            result.extend(all_subtypes(child_key))
    return result


# ═══════════════════════════════════════════════════════════════════════
# Statistics & Export
# ═══════════════════════════════════════════════════════════════════════

def print_hierarchy(node: str = None, indent: int = 0):
    """Pretty-print the operator hierarchy."""
    if node is None:
        # Print all roots
        roots = [k for k, v in HIERARCHY.items() if v.parent is None]
        for root in roots:
            print_hierarchy(root, 0)
        return
    
    if node not in HIERARCHY:
        return
    
    op = HIERARCHY[node]
    prefix = "  " * indent
    display = f"{op.name}<{op.specialization}>" if op.specialization else op.name
    n_roles = len(op.roles) + len(op.extra_roles)
    print(f"{prefix}├── {display} ({n_roles} roles) — {op.description[:60]}")
    
    for child_name in op.children:
        child_key = f"{op.name}<{child_name}>"
        print_hierarchy(child_key, indent + 1)


def export_for_tokenizer() -> Dict[str, int]:
    """
    Export all operator tags needed for the tokenizer.
    Returns a mapping of tag_string → suggested_token_id.
    
    For Hydra training, we need tokens for:
      [KN:PTRANS], [KN:PTRANS<Birth>], [KN:ATRANS], etc.
    Plus the role markers:
      [KN:agent=], [KN:theme=], [KN:source=], [KN:destination=], etc.
    """
    tags = {}
    base_id = 50282  # After existing Hydra tags (50257-50281)
    
    # Primitive tags
    for key, op in HIERARCHY.items():
        if op.specialization:
            tag = f"KN:{op.name}<{op.specialization}>"
        else:
            tag = f"KN:{op.name}"
        tags[tag] = base_id
        base_id += 1
    
    # Role tags (generic roles shared across primitives)
    all_roles = set()
    for op in HIERARCHY.values():
        all_roles.update(op.roles.keys())
        all_roles.update(op.extra_roles.keys())
    
    for role in sorted(all_roles):
        tags[f"KN:{role}="] = base_id
        base_id += 1
    
    # Special tokens
    tags["KN:OPEN"] = base_id
    base_id += 1
    tags["KN:CLOSE"] = base_id
    base_id += 1
    
    return tags


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "hierarchy":
        print_hierarchy()
    elif len(sys.argv) > 1 and sys.argv[1] == "tokens":
        tokens = export_for_tokenizer()
        print(f"\n{len(tokens)} knowledge tokens needed:")
        for tag, tid in sorted(tokens.items(), key=lambda x: x[1]):
            print(f"  {tid}: {tag}")
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Demo classification
        examples = [
            ("born_in", {"person": "Napoleon", "place": "Corsica"}, "P19"),
            ("author", {"work": "Hamlet", "author": "Shakespeare"}, "P50"),
            ("plays_for", {"player": "Messi", "team": "Inter Miami"}, "P54"),
            ("instance_of", {"entity": "Earth", "class": "planet"}, "P31"),
            ("father", {"person": "Luke", "father": "Anakin"}, "P22"),
        ]
        for rel, bindings, pid in examples:
            prim, spec, roles = classify_relation(rel, bindings, pid)
            tag = format_fact_tag(prim, spec, rel, roles, level="full")
            print(f"  {rel:15s} → {prim}<{spec}>  →  {tag}")
    else:
        print("Usage: python3 operator_hierarchy.py [hierarchy|tokens|demo]")
        print()
        print(f"Hierarchy: {len(HIERARCHY)} operator types")
        n_transfer = sum(1 for v in HIERARCHY.values() if v.parent and "TRANS" in (v.parent or ""))
        n_structural = sum(1 for v in HIERARCHY.values() if v.name in ("IS_A", "PARTOF", "QUANT"))
        print(f"  Transfer-based: {n_transfer}")
        print(f"  Structural: {n_structural}")
        print(f"  Wikidata mappings: {len(WIKIDATA_MAP)}")
