#!/usr/bin/env python3
"""
Build Knowledge Store for Hydra Training
==========================================
Populates a lambda-store-format knowledge base with facts from Wikidata,
ConceptNet, and ATOMIC, targeted at entities appearing in WikiText-103.

Pipeline:
  1. Extract article titles from raw WikiText-103 (= article = markers)
  2. Extract named entities from training sentences (spaCy NER)
  3. Fetch structured facts from Wikidata API for top entities
  4. Convert to lambda store assertion format
  5. Build sentence→fact retrieval index (TF-IDF entity matching)
  6. Augment training JSONL with [FACT:...] tokens

Output: /data/hydra/knowledge_store/
  - entities.json          (entity inventory with Wikidata IDs)
  - facts.jsonl            (all facts in lambda assertion format)
  - store.json             (serialized modal lambda store)
  - sentence_facts.jsonl   (sentence_id → relevant fact_ids mapping)

Usage:
  python3 build_knowledge_store.py --step extract-entities
  python3 build_knowledge_store.py --step fetch-wikidata
  python3 build_knowledge_store.py --step build-store
  python3 build_knowledge_store.py --step index
  python3 build_knowledge_store.py --step augment
  python3 build_knowledge_store.py --step all
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Config ─────────────────────────────────────────────────────────────

DATA_DIR = Path("/data/hydra/phase05_full_v2")
STORE_DIR = Path("/data/hydra/knowledge_store")
STORE_DIR.mkdir(parents=True, exist_ok=True)

WIKITEXT_CACHE = Path("/data/hydra/wikitext103_cache")
WIKITEXT_CACHE.mkdir(parents=True, exist_ok=True)

# Wikidata API
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Limits
MAX_ENTITIES = 5000        # Top entities by frequency
MAX_FACTS_PER_ENTITY = 20  # Cap facts per entity
WIKIDATA_BATCH = 50        # Entities per Wikidata API call
RATE_LIMIT_SEC = 1.0       # Between API calls

# Property mappings: Wikidata property → lambda operator + role names
# These map structured Wikidata relations to our operator vocabulary
PROPERTY_MAP = {
    # Identity / Classification
    "P31":   ("instance_of",    {"entity": None, "class": "value"}),
    "P279":  ("subclass_of",    {"entity": None, "class": "value"}),
    "P361":  ("part_of",        {"part": None, "whole": "value"}),
    "P527":  ("has_part",       {"whole": None, "part": "value"}),
    
    # People
    "P19":   ("born_in",        {"person": None, "place": "value"}),
    "P20":   ("died_in",        {"person": None, "place": "value"}),
    "P569":  ("born_date",      {"person": None, "date": "value"}),
    "P570":  ("died_date",      {"person": None, "date": "value"}),
    "P27":   ("citizen_of",     {"person": None, "country": "value"}),
    "P106":  ("occupation",     {"person": None, "role": "value"}),
    "P22":   ("father",         {"person": None, "father": "value"}),
    "P25":   ("mother",         {"person": None, "mother": "value"}),
    "P26":   ("spouse",         {"person": None, "spouse": "value"}),
    "P40":   ("child",          {"parent": None, "child": "value"}),
    "P69":   ("educated_at",    {"person": None, "institution": "value"}),
    "P108":  ("employer",       {"person": None, "employer": "value"}),
    
    # Geography
    "P17":   ("country",        {"entity": None, "country": "value"}),
    "P131":  ("located_in",     {"entity": None, "location": "value"}),
    "P36":   ("capital",        {"entity": None, "capital": "value"}),
    "P30":   ("continent",      {"entity": None, "continent": "value"}),
    "P625":  ("coordinates",    {"entity": None, "coords": "value"}),
    "P1082": ("population",     {"entity": None, "population": "value"}),
    
    # Creative works
    "P50":   ("author",         {"work": None, "author": "value"}),
    "P57":   ("director",       {"work": None, "director": "value"}),
    "P86":   ("composer",       {"work": None, "composer": "value"}),
    "P161":  ("cast_member",    {"work": None, "actor": "value"}),
    "P136":  ("genre",          {"work": None, "genre": "value"}),
    "P577":  ("pub_date",       {"work": None, "date": "value"}),
    "P495":  ("country_origin", {"work": None, "country": "value"}),
    "P407":  ("language",       {"work": None, "language": "value"}),
    
    # Organizations
    "P112":  ("founded_by",     {"org": None, "founder": "value"}),
    "P571":  ("founded_date",   {"entity": None, "date": "value"}),
    "P159":  ("headquarters",   {"org": None, "location": "value"}),
    "P169":  ("ceo",            {"org": None, "person": "value"}),
    "P452":  ("industry",       {"org": None, "industry": "value"}),
    
    # Science / Nature
    "P171":  ("parent_taxon",   {"entity": None, "taxon": "value"}),
    "P186":  ("material",       {"entity": None, "material": "value"}),
    "P1376": ("capital_of",     {"city": None, "territory": "value"}),
    
    # Sports
    "P54":   ("plays_for",      {"player": None, "team": "value"}),
    "P413":  ("position",       {"player": None, "position": "value"}),
    "P118":  ("league",         {"team": None, "league": "value"}),
}


# ── Step 1: Extract Entities ──────────────────────────────────────────

def extract_entities():
    """Extract entities from WikiText-103 via HuggingFace datasets + NER."""
    print("=" * 60)
    print("Step 1: Extracting entities from WikiText-103")
    print("=" * 60)
    
    # Get article titles from raw WikiText-103
    print("\nLoading WikiText-103 raw...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=str(WIKITEXT_CACHE))
    
    # Extract article titles (lines starting with " = Title = \n")
    article_titles = []
    title_pattern = re.compile(r'^\s*= ([^=]+) =\s*$')
    
    for split in ["train", "validation"]:
        for line in ds[split]["text"]:
            m = title_pattern.match(line)
            if m:
                title = m.group(1).strip()
                if title and len(title) > 1:
                    article_titles.append(title)
    
    print(f"  Found {len(article_titles)} article titles")
    
    # Also extract named entities from training sentences using regex
    # (faster than spaCy for this scale — we can refine later)
    print("\nExtracting named entities from training data...")
    entity_counts = Counter()
    
    # Add article titles
    for t in article_titles:
        entity_counts[t] += 10  # boost article titles
    
    # Scan training JSONL
    train_path = DATA_DIR / "hydra_train.jsonl"
    n_lines = 0
    
    with open(train_path) as f:
        for line in f:
            d = json.loads(line)
            raw = d.get("raw", "")
            
            # Multi-word capitalized phrases (crude NER)
            for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+(?:of|the|and|in|de|von|van|for)\s+)?(?:[A-Z][a-z]+\s*)+)', raw):
                ent = m.group().strip()
                if len(ent) > 2 and not ent.startswith("The "):
                    entity_counts[ent] += 1
            
            # Single capitalized words that aren't sentence-initial
            words = raw.split()
            for i, w in enumerate(words):
                if i > 0 and w[0].isupper() and w.isalpha() and len(w) > 3:
                    entity_counts[w] += 1
            
            n_lines += 1
            if n_lines % 500000 == 0:
                print(f"  Scanned {n_lines:,} sentences, {len(entity_counts):,} unique entities")
    
    print(f"  Total: {n_lines:,} sentences, {len(entity_counts):,} unique entities")
    
    # Filter and rank
    # Remove very common English words that get false-positived
    stopwords = {
        "The", "This", "That", "These", "Those", "There", "Their", "They",
        "What", "When", "Where", "Which", "While", "With", "Within",
        "After", "Before", "During", "However", "Although", "According",
        "Also", "Another", "Both", "Each", "Every", "Many", "Most",
        "Other", "Some", "Such", "Very", "About", "Over", "Under",
        "Between", "Into", "Through", "From", "More", "Less", "Only",
        "First", "Second", "Third", "Last", "Next", "Following",
        "January", "February", "March", "April", "June", "July",
        "August", "September", "October", "November", "December",
    }
    
    for sw in stopwords:
        entity_counts.pop(sw, None)
    
    # Take top N entities
    top_entities = entity_counts.most_common(MAX_ENTITIES)
    
    print(f"\nTop 30 entities:")
    for ent, count in top_entities[:30]:
        print(f"  {count:6d}  {ent}")
    
    # Save
    entities_out = {
        "entities": [
            {"name": name, "count": count, "wikidata_id": None}
            for name, count in top_entities
        ],
        "total_unique": len(entity_counts),
        "total_sentences": n_lines,
    }
    
    out_path = STORE_DIR / "entities.json"
    with open(out_path, "w") as f:
        json.dump(entities_out, f, indent=2)
    
    print(f"\nSaved {len(top_entities)} entities to {out_path}")
    return entities_out


# ── Step 2: Fetch Wikidata ────────────────────────────────────────────

def fetch_wikidata():
    """Resolve entity names to Wikidata IDs and fetch structured facts."""
    print("=" * 60)
    print("Step 2: Fetching facts from Wikidata")
    print("=" * 60)
    
    # Load entities
    entities_path = STORE_DIR / "entities.json"
    with open(entities_path) as f:
        data = json.load(f)
    entities = data["entities"]
    
    # Phase 1: Resolve names to Wikidata IDs
    print(f"\nResolving {len(entities)} entity names...")
    resolved = 0
    facts_file = STORE_DIR / "facts_raw.jsonl"
    
    # Check for resume
    already_done = set()
    if facts_file.exists():
        with open(facts_file) as f:
            for line in f:
                d = json.loads(line)
                already_done.add(d.get("entity_name"))
        print(f"  Resuming: {len(already_done)} entities already done")
    
    fout = open(facts_file, "a")
    
    for i in range(0, len(entities), WIKIDATA_BATCH):
        batch = entities[i : i + WIKIDATA_BATCH]
        batch = [e for e in batch if e["name"] not in already_done]
        
        if not batch:
            continue
        
        for ent in batch:
            name = ent["name"]
            try:
                # Search Wikidata for entity
                wid = _wikidata_search(name)
                if not wid:
                    continue
                
                # Fetch claims for this entity
                facts = _wikidata_get_claims(wid, name)
                
                for fact in facts:
                    fout.write(json.dumps(fact) + "\n")
                
                resolved += 1
                ent["wikidata_id"] = wid
                
                if resolved % 50 == 0:
                    print(f"  Resolved {resolved} entities, latest: {name} → {wid}")
                    fout.flush()
                
            except Exception as e:
                print(f"  Error for {name}: {e}")
            
            time.sleep(RATE_LIMIT_SEC)
    
    fout.close()
    
    # Update entities with Wikidata IDs
    with open(entities_path, "w") as f:
        json.dump(data, f, indent=2)
    
    # Count facts
    n_facts = sum(1 for _ in open(facts_file))
    print(f"\nDone: {resolved} entities resolved, {n_facts} facts collected")


def _wikidata_search(name: str) -> Optional[str]:
    """Search Wikidata for an entity by name, return QID."""
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "limit": 1,
        "format": "json",
    }
    url = WIKIDATA_API + "?" + urllib.parse.urlencode(params)
    
    req = urllib.request.Request(url, headers={
        "User-Agent": "HydraKnowledgeStore/1.0 (palexanderbalogh@gmail.com)"
    })
    
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    
    results = data.get("search", [])
    if results:
        return results[0]["id"]
    return None


def _wikidata_get_claims(qid: str, entity_name: str) -> List[dict]:
    """Fetch and parse claims for a Wikidata entity."""
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "claims|labels|descriptions",
        "languages": "en",
        "format": "json",
    }
    url = WIKIDATA_API + "?" + urllib.parse.urlencode(params)
    
    req = urllib.request.Request(url, headers={
        "User-Agent": "HydraKnowledgeStore/1.0 (palexanderbalogh@gmail.com)"
    })
    
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    
    entity_data = data.get("entities", {}).get(qid, {})
    claims = entity_data.get("claims", {})
    
    # Get label
    labels = entity_data.get("labels", {})
    label = labels.get("en", {}).get("value", entity_name)
    
    facts = []
    
    for prop_id, claim_list in claims.items():
        if prop_id not in PROPERTY_MAP:
            continue
        
        operator, role_template = PROPERTY_MAP[prop_id]
        
        for claim in claim_list[:3]:  # Max 3 values per property
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            
            value = _extract_value(datavalue)
            if not value:
                continue
            
            # Build bindings
            bindings = {}
            for role, source in role_template.items():
                if source is None:
                    bindings[role] = label
                elif source == "value":
                    bindings[role] = value
                else:
                    bindings[role] = source
            
            facts.append({
                "entity_name": entity_name,
                "entity_id": qid,
                "operator": operator,
                "bindings": bindings,
                "property_id": prop_id,
                "world": "ACTUAL",
            })
            
            if len(facts) >= MAX_FACTS_PER_ENTITY:
                return facts
    
    return facts


def _extract_value(datavalue: dict) -> Optional[str]:
    """Extract a human-readable value from Wikidata datavalue."""
    vtype = datavalue.get("type")
    value = datavalue.get("value")
    
    if not value:
        return None
    
    if vtype == "wikibase-entityid":
        # Need to resolve the QID to a label
        qid = value.get("id")
        if qid:
            return _resolve_label(qid)
    
    elif vtype == "string":
        return value
    
    elif vtype == "time":
        # Parse Wikidata time format: "+1879-03-14T00:00:00Z"
        time_str = value.get("time", "")
        m = re.match(r'[+-]?(\d{4})-(\d{2})-(\d{2})', time_str)
        if m:
            y, mo, d = m.groups()
            if mo == "00":
                return y
            elif d == "00":
                return f"{y}-{mo}"
            return f"{y}-{mo}-{d}"
    
    elif vtype == "quantity":
        amount = value.get("amount", "")
        return amount.lstrip("+")
    
    elif vtype == "monolingualtext":
        return value.get("text")
    
    elif vtype == "globecoordinate":
        lat = value.get("latitude")
        lon = value.get("longitude")
        if lat is not None and lon is not None:
            return f"{lat:.2f},{lon:.2f}"
    
    return None


_label_cache = {}

def _resolve_label(qid: str) -> Optional[str]:
    """Resolve a Wikidata QID to its English label."""
    if qid in _label_cache:
        return _label_cache[qid]
    
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels",
        "languages": "en",
        "format": "json",
    }
    url = WIKIDATA_API + "?" + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "HydraKnowledgeStore/1.0 (palexanderbalogh@gmail.com)"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        
        label = data["entities"][qid]["labels"].get("en", {}).get("value")
        _label_cache[qid] = label
        time.sleep(0.2)  # gentle rate limit
        return label
    except Exception:
        _label_cache[qid] = None
        return None


# ── Step 3: Build Lambda Store (with Operator Hierarchy) ──────────────

def build_store():
    """Convert raw facts into a structured lambda store with Schankian primitives."""
    print("=" * 60)
    print("Step 3: Building Unified Knowledge Store")
    print("  (surface relations + Schankian primitives + hierarchy)")
    print("=" * 60)
    
    # Import operator hierarchy
    sys.path.insert(0, str(Path(__file__).parent))
    from operator_hierarchy import classify_relation, WIKIDATA_MAP, HIERARCHY, format_fact_tag
    
    facts_path = STORE_DIR / "facts_raw.jsonl"
    
    # Read and deduplicate facts
    facts = []
    seen = set()
    
    with open(facts_path) as f:
        for line in f:
            d = json.loads(line)
            key = (d["operator"], json.dumps(d["bindings"], sort_keys=True))
            if key not in seen:
                seen.add(key)
                facts.append(d)
    
    print(f"  {len(facts)} unique facts from {sum(1 for _ in open(facts_path))} raw")
    
    # Classify each fact through the operator hierarchy
    primitive_counts = Counter()
    spec_counts = Counter()
    unclassified = 0
    
    store_facts = []
    for i, f in enumerate(facts):
        # Get Wikidata property ID
        prop_id = f.get("property_id")
        surface_op = f["operator"]
        bindings = f["bindings"]
        
        # Classify through hierarchy
        primitive, specialization, unified_roles = classify_relation(
            surface_op, bindings, prop_id
        )
        
        if primitive == "TRANSFER" and specialization == "Unknown":
            unclassified += 1
        
        primitive_counts[primitive] += 1
        spec_counts[f"{primitive}<{specialization}>"] += 1
        
        # Generate tags at all four levels
        tags = {
            "surface":   format_fact_tag(primitive, specialization, surface_op, 
                                         unified_roles if unified_roles else bindings, "surface"),
            "primitive": format_fact_tag(primitive, specialization, surface_op,
                                         unified_roles if unified_roles else bindings, "primitive"),
            "full":      format_fact_tag(primitive, specialization, surface_op,
                                         unified_roles if unified_roles else bindings, "full"),
            "generic":   format_fact_tag(primitive, specialization, surface_op,
                                         unified_roles if unified_roles else bindings, "generic"),
        }
        
        store_facts.append({
            "fact_id": i,
            "surface_operator": surface_op,
            "primitive": primitive,
            "specialization": specialization,
            "bindings": bindings,
            "unified_roles": unified_roles if unified_roles else bindings,
            "tags": tags,
            "property_id": prop_id,
            "world": f.get("world", "ACTUAL"),
            "source": f"wikidata:{f.get('entity_id', 'unknown')}",
        })
    
    # Report
    print(f"\n  Primitive distribution:")
    for prim, count in primitive_counts.most_common():
        print(f"    {prim:12s}: {count:6d}")
    
    print(f"\n  Top 20 specializations:")
    for spec, count in spec_counts.most_common(20):
        print(f"    {spec:30s}: {count:6d}")
    
    if unclassified:
        print(f"\n  ⚠ {unclassified} facts could not be classified (TRANSFER<Unknown>)")
    
    # Build entity inventory
    all_entities = set()
    for f in store_facts:
        for v in f["bindings"].values():
            if v:
                all_entities.add(v)
        for v in f["unified_roles"].values():
            if v:
                all_entities.add(v)
    
    print(f"\n  {len(all_entities)} unique entities across all facts")
    
    # Save enriched facts
    out_path = STORE_DIR / "facts.jsonl"
    with open(out_path, "w") as f:
        for fact in store_facts:
            f.write(json.dumps(fact) + "\n")
    
    # Save entity inventory
    entity_list = sorted(all_entities)
    entity_to_id = {e: i for i, e in enumerate(entity_list)}
    with open(STORE_DIR / "entity_vocab.json", "w") as f:
        json.dump({
            "entities": entity_list,
            "entity_to_id": entity_to_id,
            "n_entities": len(entity_list),
        }, f, indent=2)
    
    # Save operator inventory (both surface and primitive levels)
    surface_ops = sorted(set(f["surface_operator"] for f in store_facts))
    primitives = sorted(primitive_counts.keys())
    specializations = sorted(spec_counts.keys())
    
    with open(STORE_DIR / "operator_vocab.json", "w") as f:
        json.dump({
            "surface_operators": surface_ops,
            "primitives": primitives,
            "specializations": specializations,
            "n_surface": len(surface_ops),
            "n_primitives": len(primitives),
            "n_specializations": len(specializations),
            "hierarchy_size": len(HIERARCHY),
        }, f, indent=2)
    
    # Save example tags for inspection
    with open(STORE_DIR / "example_tags.txt", "w") as f:
        f.write("# Example fact tags at all four levels\n")
        f.write("# (first 50 facts)\n\n")
        for fact in store_facts[:50]:
            f.write(f"Surface:   {fact['tags']['surface']}\n")
            f.write(f"Primitive: {fact['tags']['primitive']}\n")
            f.write(f"Full:      {fact['tags']['full']}\n")
            f.write(f"Generic:   {fact['tags']['generic']}\n")
            f.write("\n")
    
    print(f"\nSaved {len(store_facts)} classified facts to {out_path}")
    print(f"Entity vocab: {len(entity_list)}")
    print(f"Operator vocab: {len(surface_ops)} surface, {len(primitives)} primitives, {len(specializations)} specializations")


# ── Step 4: Build Sentence → Fact Index ───────────────────────────────

def build_index():
    """Build an index mapping each training sentence to relevant facts."""
    print("=" * 60)
    print("Step 4: Building sentence → fact retrieval index")
    print("=" * 60)
    
    # Load facts
    facts = []
    with open(STORE_DIR / "facts.jsonl") as f:
        for line in f:
            facts.append(json.loads(line))
    
    print(f"  Loaded {len(facts)} facts")
    
    # Build entity → fact_ids index
    entity_to_facts = defaultdict(list)
    for fact in facts:
        for role, entity in fact["bindings"].items():
            if entity:
                # Index by normalized name and substrings
                entity_lower = entity.lower()
                entity_to_facts[entity_lower].append(fact["fact_id"])
                # Also index by last name for people
                parts = entity.split()
                if len(parts) > 1:
                    entity_to_facts[parts[-1].lower()].append(fact["fact_id"])
    
    print(f"  Entity index: {len(entity_to_facts)} keys")
    
    # Scan training data and map sentences to facts
    print("\n  Matching sentences to facts...")
    train_path = DATA_DIR / "hydra_train.jsonl"
    out_path = STORE_DIR / "sentence_facts.jsonl"
    
    n_matched = 0
    n_total = 0
    
    with open(train_path) as fin, open(out_path, "w") as fout:
        for line_no, line in enumerate(fin):
            d = json.loads(line)
            raw = d.get("raw", "").lower()
            
            # Find matching facts
            matched_fact_ids = set()
            for entity_key, fact_ids in entity_to_facts.items():
                if entity_key in raw and len(entity_key) > 3:
                    matched_fact_ids.update(fact_ids)
            
            if matched_fact_ids:
                # Limit to top 5 most relevant facts
                fact_ids = sorted(matched_fact_ids)[:5]
                fout.write(json.dumps({
                    "sentence_id": line_no,
                    "fact_ids": fact_ids,
                }) + "\n")
                n_matched += 1
            
            n_total += 1
            if n_total % 500000 == 0:
                print(f"    {n_total:,} sentences, {n_matched:,} matched ({100*n_matched/n_total:.1f}%)")
    
    print(f"\n  Done: {n_matched:,} / {n_total:,} sentences matched ({100*n_matched/n_total:.1f}%)")


# ── Step 5: Augment Training Data ─────────────────────────────────────

def augment_data():
    """Inject hierarchical fact tokens into training JSONL."""
    print("=" * 60)
    print("Step 5: Augmenting training data with hierarchical knowledge facts")
    print("=" * 60)
    
    # Load facts (now with hierarchy classification)
    facts = []
    with open(STORE_DIR / "facts.jsonl") as f:
        for line in f:
            facts.append(json.loads(line))
    
    # Load sentence→fact mapping
    sentence_facts = {}
    with open(STORE_DIR / "sentence_facts.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sentence_facts[d["sentence_id"]] = d["fact_ids"]
    
    print(f"  {len(facts)} facts, {len(sentence_facts)} sentences with facts")
    
    # Tag level: use "full" by default (PTRANS<Birth>/born_in)
    # Also generate all levels so we can ablate later
    TAG_LEVEL = "full"
    
    # Process training data
    train_in = DATA_DIR / "hydra_train.jsonl"
    train_out = STORE_DIR / "hydra_train_augmented.jsonl"
    
    n_augmented = 0
    n_total = 0
    n_fact_tokens = 0
    primitive_usage = Counter()
    
    with open(train_in) as fin, open(train_out, "w") as fout:
        for line_no, line in enumerate(fin):
            d = json.loads(line)
            
            if line_no in sentence_facts:
                fact_ids = sentence_facts[line_no]
                fact_tags_by_level = {"surface": [], "primitive": [], "full": [], "generic": []}
                
                for fid in fact_ids:
                    if fid < len(facts):
                        fact = facts[fid]
                        # Use pre-computed tags from the hierarchy
                        if "tags" in fact:
                            for level in fact_tags_by_level:
                                fact_tags_by_level[level].append(fact["tags"].get(level, ""))
                            primitive_usage[fact.get("primitive", "?")] += 1
                        else:
                            # Fallback for old-format facts
                            bindings_str = " ".join(
                                f"{k}={v}" for k, v in fact["bindings"].items() if v
                            )
                            tag = f"[KN:{fact.get('operator', '?')} {bindings_str}]"
                            for level in fact_tags_by_level:
                                fact_tags_by_level[level].append(tag)
                
                # Store all levels (for ablation experiments)
                d["knowledge_tags"] = fact_tags_by_level[TAG_LEVEL]
                d["knowledge_tags_all"] = fact_tags_by_level
                d["n_facts"] = len(fact_tags_by_level[TAG_LEVEL])
                n_augmented += 1
                n_fact_tokens += len(fact_tags_by_level[TAG_LEVEL])
            else:
                d["knowledge_tags"] = []
                d["n_facts"] = 0
            
            fout.write(json.dumps(d) + "\n")
            n_total += 1
            
            if n_total % 500000 == 0:
                print(f"    {n_total:,} sentences, {n_augmented:,} augmented")
    
    print(f"\n  Done: {n_augmented:,} / {n_total:,} sentences augmented")
    print(f"  Total fact tokens injected: {n_fact_tokens:,}")
    print(f"  Average facts per augmented sentence: {n_fact_tokens/max(n_augmented,1):.1f}")
    print(f"  Default tag level: {TAG_LEVEL}")
    
    print(f"\n  Primitive usage in matched sentences:")
    for prim, count in primitive_usage.most_common():
        print(f"    {prim:12s}: {count:6d}")
    
    # Also augment validation data
    val_in = DATA_DIR / "hydra_val.jsonl"
    val_out = STORE_DIR / "hydra_val_augmented.jsonl"
    
    print("\n  Augmenting validation data...")
    
    # Build entity index
    entity_to_facts = defaultdict(list)
    for fact in facts:
        for role, entity in fact["bindings"].items():
            if entity:
                entity_to_facts[entity.lower()].append(fact["fact_id"])
                parts = entity.split()
                if len(parts) > 1:
                    entity_to_facts[parts[-1].lower()].append(fact["fact_id"])
    
    n_val_aug = 0
    with open(val_in) as fin, open(val_out, "w") as fout:
        for line_no, line in enumerate(fin):
            d = json.loads(line)
            raw = d.get("raw", "").lower()
            
            matched = set()
            for ek, fids in entity_to_facts.items():
                if ek in raw and len(ek) > 3:
                    matched.update(fids)
            
            fact_tags_by_level = {"surface": [], "primitive": [], "full": [], "generic": []}
            for fid in sorted(matched)[:5]:
                if fid < len(facts):
                    fact = facts[fid]
                    if "tags" in fact:
                        for level in fact_tags_by_level:
                            fact_tags_by_level[level].append(fact["tags"].get(level, ""))
            
            d["knowledge_tags"] = fact_tags_by_level[TAG_LEVEL]
            d["knowledge_tags_all"] = fact_tags_by_level
            d["n_facts"] = len(fact_tags_by_level[TAG_LEVEL])
            if fact_tags_by_level[TAG_LEVEL]:
                n_val_aug += 1
            
            fout.write(json.dumps(d) + "\n")
    
    print(f"  Validation: {n_val_aug} sentences augmented")
    
    # Summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Facts in store:        {len(facts):,}")
    print(f"  Training augmented:    {n_augmented:,} / {n_total:,}")
    print(f"  Fact tokens added:     {n_fact_tokens:,}")
    print(f"  Tag level:             {TAG_LEVEL}")
    print(f"  All levels stored:     surface, primitive, full, generic")
    print(f"  Output train:          {train_out}")
    print(f"  Output val:            {val_out}")
    print(f"\nNext: pretokenize the augmented data with knowledge tags")


# ── Main ───────────────────────────────────────────────────────────────

STEPS = {
    "extract-entities": extract_entities,
    "fetch-wikidata": fetch_wikidata,
    "build-store": build_store,
    "index": build_index,
    "augment": augment_data,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=list(STEPS.keys()) + ["all"], default="all")
    args = parser.parse_args()
    
    if args.step == "all":
        for name, fn in STEPS.items():
            print(f"\n\n{'#'*60}")
            print(f"# Running: {name}")
            print(f"{'#'*60}\n")
            fn()
    else:
        STEPS[args.step]()


if __name__ == "__main__":
    main()
