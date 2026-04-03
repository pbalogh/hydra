#!/usr/bin/env python3
"""
Lambda Bridge: Shadow memory system for the agent.

Shadows the existing markdown-based memory with a structured lambda store.
Does NOT replace or modify the markdown system — runs alongside it.

Usage:
    python3 lambda_bridge.py assert <operator> key=value key=value ...
    python3 lambda_bridge.py query [operator] [key=value ...]
    python3 lambda_bridge.py history [entity]
    python3 lambda_bridge.py export
    python3 lambda_bridge.py stats
    python3 lambda_bridge.py repl

The store persists to ~/clawd/memory/lambda_store.json
Exports to ~/clawd/memory/lambda_export.md (indexed by memory_search)

Operator conventions:
    project_emerged   - new project/concept named (entity, name, concept, date)
    project_status    - status update (entity, status, detail, date)
    project_pivot     - pivot from one thing to another (from, to, reason, date)
    has_component     - project has a component (project, component, role)
    decision          - a decision was made (topic, choice, reason, date)
    idea              - an idea emerged (entity, description, context, date)
    connection        - two things are connected (a, b, relationship)
    paper_link        - a paper is relevant (paper, project, relevance)
    person_said       - someone said something notable (person, claim, context, date)
    architecture      - architecture detail (project, component, detail)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the lambda store code to path
STORE_CODE = Path.home() / "clawd/projects/mentalese-grammar/code"
sys.path.insert(0, str(STORE_CODE))

from modal_lambda_store import ModalLambdaStore, Modality

STORE_PATH = Path.home() / "clawd/memory/lambda_store.json"
EXPORT_PATH = Path.home() / "clawd/memory/lambda_export.md"
RECALL_LOG = Path.home() / "clawd/memory/recall_log.jsonl"


def load_store() -> ModalLambdaStore:
    """Load the persistent store, or create a new one."""
    if STORE_PATH.exists():
        return ModalLambdaStore.load(str(STORE_PATH))
    return ModalLambdaStore()


def save_store(store: ModalLambdaStore):
    """Save the store and auto-export markdown."""
    store.persist(str(STORE_PATH))
    export_markdown(store)


def parse_bindings(args: list) -> dict:
    """Parse key=value pairs from command line args."""
    bindings = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            bindings[key] = value
    return bindings


def cmd_assert(args: list):
    """Assert a fact into the store."""
    if not args:
        print("Usage: assert <operator> key=value key=value ...")
        print("Example: assert project_emerged entity=hydra name=Hydra concept='multi-head architecture' date=2026-04-02")
        sys.exit(1)

    operator = args[0]
    bindings = parse_bindings(args[1:])

    # Auto-add timestamp if not present and date not specified
    if 'date' not in bindings and 'timestamp' not in bindings:
        bindings['date'] = datetime.now().strftime('%Y-%m-%d')

    store = load_store()
    event = store.assert_fact(operator, **bindings)
    save_store(store)

    print(f"✓ Asserted [{operator}] {bindings}")
    print(f"  Event #{event.event_id} in world {event.world}")


def cmd_query(args: list):
    """Query the store."""
    operator = None
    bindings = {}

    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            bindings[key] = value
        elif operator is None:
            operator = arg

    store = load_store()

    if operator:
        results = store.query(operator, **bindings)
    elif bindings:
        results = store.query(**bindings)
    else:
        # Show everything
        results = store.query()

    # Log the recall attempt
    _log_recall(
        query_type='structured',
        query={'operator': operator, **bindings},
        num_results=len(results),
        result_ids=[r.event_id for r in results],
    )

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} result(s):\n")
    for r in results:
        print(f"  [{r.operator}] {_fmt_bindings(r.bindings)}")


def cmd_search(args: list):
    """Fuzzy search across all fact values and operators.
    
    Unlike 'query' (strict key=value matching), search finds any fact
    where the search term appears in ANY field — operator names, keys, or values.
    This is the "I vaguely remember something about X" command.
    """
    if not args:
        print("Usage: search <term> [term2 ...]")
        print("  Searches all fact values, operators, and keys for matching terms.")
        print("  Multiple terms = AND (all must match)")
        return

    store = load_store()
    all_events = store.query()
    terms = [t.lower() for t in args]

    results = []
    for e in all_events:
        # Build a searchable string from everything about this fact
        searchable = e.operator.lower() + ' ' + ' '.join(
            f"{k} {v}".lower() for k, v in e.bindings.items()
        )
        if all(term in searchable for term in terms):
            results.append(e)

    # Log the recall attempt
    _log_recall(
        query_type='fuzzy',
        query={'terms': args},
        num_results=len(results),
        result_ids=[r.event_id for r in results],
    )

    if not results:
        print(f"No results matching: {' AND '.join(args)}")
        return

    print(f"Found {len(results)} result(s) matching '{' AND '.join(args)}':\n")
    for r in results:
        # Highlight which fields matched
        print(f"  #{r.event_id} [{r.operator}]")
        for k, v in r.bindings.items():
            # Bold the matching parts
            marker = ' ◀' if any(t in f"{k} {v}".lower() for t in terms) else ''
            print(f"    {k}: {v}{marker}")
        print()


def cmd_history(args: list):
    """Show history for an entity or all events."""
    store = load_store()

    if args:
        entity = args[0]
        # Search broadly: any event where the entity appears in any binding value or operator
        all_events = store.query()
        results = [e for e in all_events
                   if entity.lower() in str(e.bindings).lower()
                   or entity.lower() in e.operator.lower()]

        if not results:
            print(f"No history found for '{entity}'")
            return

        print(f"History for '{entity}':\n")
        for r in sorted(results, key=lambda e: e.event_id):
            date = r.bindings.get('date', '?')
            print(f"  #{r.event_id} [{r.operator}] {date}")
            for k, v in r.bindings.items():
                if k != 'date':
                    print(f"    {k}: {v}")
            print()
    else:
        # All events chronologically
        all_events = store.query()
        print(f"All {len(all_events)} events:\n")
        for r in sorted(all_events, key=lambda e: e.event_id):
            print(f"  #{r.event_id} [{r.operator}] {_fmt_bindings(r.bindings)}")


def cmd_export(args: list):
    """Export the store to markdown."""
    store = load_store()
    export_markdown(store)
    print(f"Exported to {EXPORT_PATH}")


def cmd_stats(args: list):
    """Show store statistics."""
    store = load_store()
    stats = store.stats()
    print(json.dumps(stats, indent=2))


def cmd_repl(args: list):
    """Interactive REPL for the store."""
    store = load_store()
    print("Lambda Bridge REPL (type 'help' for commands, 'quit' to exit)")
    print(f"Store: {store.stats()['total_events']} events\n")

    while True:
        try:
            line = input("λ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ('quit', 'exit', 'q'):
            break
        if line == 'help':
            print("  assert <op> k=v ...  — add a fact")
            print("  query [op] [k=v ...] — search facts")
            print("  history [entity]     — show history")
            print("  export               — export to markdown")
            print("  stats                — show statistics")
            print("  quit                 — exit")
            continue

        parts = line.split()
        cmd = parts[0]
        rest = parts[1:]

        if cmd == 'assert':
            if not rest:
                print("  Usage: assert <operator> key=value ...")
                continue
            operator = rest[0]
            bindings = parse_bindings(rest[1:])
            if 'date' not in bindings:
                bindings['date'] = datetime.now().strftime('%Y-%m-%d')
            event = store.assert_fact(operator, **bindings)
            print(f"  ✓ [{operator}] {_fmt_bindings(bindings)}")

        elif cmd == 'query':
            operator = None
            bindings = {}
            for arg in rest:
                if '=' in arg:
                    k, v = arg.split('=', 1)
                    bindings[k] = v
                elif operator is None:
                    operator = arg
            if operator:
                results = store.query(operator, **bindings)
            elif bindings:
                results = store.query(**bindings)
            else:
                results = store.query()
            for r in results:
                print(f"  [{r.operator}] {_fmt_bindings(r.bindings)}")
            if not results:
                print("  (no results)")

        elif cmd == 'history':
            entity = rest[0] if rest else None
            if entity:
                results = store.query(entity=entity)
                if not results:
                    all_events = store.query()
                    results = [e for e in all_events
                               if entity.lower() in str(e.bindings).lower()]
                for r in results:
                    print(f"  #{r.event_id} [{r.operator}] {_fmt_bindings(r.bindings)}")
            else:
                for r in store.query():
                    print(f"  #{r.event_id} [{r.operator}] {_fmt_bindings(r.bindings)}")

        elif cmd == 'export':
            export_markdown(store)
            print(f"  Exported to {EXPORT_PATH}")

        elif cmd == 'stats':
            print(f"  {store.stats()}")

        else:
            print(f"  Unknown command: {cmd}")

    save_store(store)
    print("Store saved.")


def _log_recall(query_type: str, query: dict, num_results: int,
                 result_ids: list, outcome: str = 'unknown', note: str = ''):
    """Log a recall attempt for later analysis.
    
    outcome: 'hit' (found what was needed), 'miss' (didn't find it),
             'partial' (found related but not exact), 'unknown' (not scored yet)
    """
    entry = {
        'timestamp': datetime.now().isoformat(),
        'query_type': query_type,
        'query': query,
        'num_results': num_results,
        'result_ids': result_ids,
        'outcome': outcome,
        'note': note,
    }
    with open(RECALL_LOG, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def cmd_recall_score(args: list):
    """Score the last recall attempt. Usage: recall-score hit|miss|partial [note]"""
    if not args:
        print("Usage: recall-score hit|miss|partial [optional note]")
        print("  Scores the most recent recall attempt in recall_log.jsonl")
        return

    outcome = args[0]
    if outcome not in ('hit', 'miss', 'partial'):
        print(f"Invalid outcome '{outcome}'. Use: hit, miss, partial")
        return

    note = ' '.join(args[1:]) if len(args) > 1 else ''

    if not RECALL_LOG.exists():
        print("No recall log found.")
        return

    # Read all entries, update the last one
    lines = RECALL_LOG.read_text().strip().split('\n')
    if not lines:
        print("Recall log is empty.")
        return

    last_entry = json.loads(lines[-1])
    last_entry['outcome'] = outcome
    if note:
        last_entry['note'] = note
    lines[-1] = json.dumps(last_entry)

    RECALL_LOG.write_text('\n'.join(lines) + '\n')
    print(f"✓ Scored last recall as '{outcome}'" + (f" ({note})" if note else ""))


def cmd_recall_stats(args: list):
    """Show recall performance statistics."""
    if not RECALL_LOG.exists():
        print("No recall log found yet. Start querying to generate data!")
        return

    entries = []
    for line in RECALL_LOG.read_text().strip().split('\n'):
        if line:
            entries.append(json.loads(line))

    total = len(entries)
    scored = [e for e in entries if e['outcome'] != 'unknown']
    hits = [e for e in entries if e['outcome'] == 'hit']
    misses = [e for e in entries if e['outcome'] == 'miss']
    partials = [e for e in entries if e['outcome'] == 'partial']
    empty = [e for e in entries if e['num_results'] == 0]

    print(f"Recall Performance ({total} attempts):")
    print(f"  Scored:    {len(scored)}/{total}")
    if scored:
        print(f"  Hits:      {len(hits)} ({len(hits)/len(scored)*100:.0f}%)")
        print(f"  Misses:    {len(misses)} ({len(misses)/len(scored)*100:.0f}%)")
        print(f"  Partial:   {len(partials)} ({len(partials)/len(scored)*100:.0f}%)")
    print(f"  Empty:     {len(empty)} (queries returning 0 results)")

    # Show recent misses (most valuable for improvement)
    if misses:
        print(f"\nRecent misses (learning opportunities):")
        for m in misses[-5:]:
            print(f"  {m['timestamp'][:16]} query={m['query']}")
            if m.get('note'):
                print(f"    note: {m['note']}")


def cmd_recall_log(args: list):
    """Show the raw recall log. Usage: recall-log [N] — show last N entries (default 10)"""
    if not RECALL_LOG.exists():
        print("No recall log found.")
        return

    n = int(args[0]) if args else 10
    entries = []
    for line in RECALL_LOG.read_text().strip().split('\n'):
        if line:
            entries.append(json.loads(line))

    for e in entries[-n:]:
        outcome_icon = {'hit': '✅', 'miss': '❌', 'partial': '🟡', 'unknown': '⬜'}.get(e['outcome'], '?')
        print(f"  {outcome_icon} {e['timestamp'][:16]} [{e['query_type']}] results={e['num_results']} query={e['query']}")
        if e.get('note'):
            print(f"     {e['note']}")


def export_markdown(store: ModalLambdaStore):
    """Export the store to a markdown file optimized for memory_search embedding."""
    all_events = store.query()
    if not all_events:
        EXPORT_PATH.write_text("# Lambda Knowledge Store\n\nNo facts stored yet.\n")
        return

    # Group by entity or project
    groups = {}  # entity_name -> list of events
    ungrouped = []

    for event in all_events:
        b = event.bindings
        # Try to find a grouping key
        group_key = (b.get('entity') or b.get('project') or
                     b.get('topic') or b.get('name'))
        if group_key:
            groups.setdefault(group_key.lower(), []).append(event)
        else:
            ungrouped.append(event)

    lines = [
        "# Lambda Knowledge Store Export",
        "",
        f"*Auto-generated from lambda_store.json — {len(all_events)} facts*",
        f"*Last export: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    # Emit grouped sections — these are optimized for embedding search
    for group_key in sorted(groups.keys()):
        events = groups[group_key]
        # Collect all metadata for this entity
        names = set()
        operators = set()
        all_values = set()
        details = []

        for e in events:
            operators.add(e.operator)
            for k, v in e.bindings.items():
                all_values.add(v)
                if k == 'name':
                    names.add(v)

            # Format as a detail line
            non_meta = {k: v for k, v in e.bindings.items()
                        if k not in ('entity', 'project', 'date')}
            date = e.bindings.get('date', '')
            detail_parts = [f"{k}={v}" for k, v in non_meta.items()]
            details.append(f"- **{e.operator}** ({date}): {', '.join(detail_parts)}")

        display_name = next(iter(names)) if names else group_key
        op_tags = ' '.join(sorted(operators))

        lines.append(f"## {display_name}")
        lines.append(f"*Tags: {op_tags}*")
        lines.append(f"*Keywords: {', '.join(sorted(all_values))}*")
        lines.append("")
        lines.extend(details)
        lines.append("")

    # Emit ungrouped facts
    if ungrouped:
        lines.append("## Other Facts")
        lines.append("")
        for e in ungrouped:
            detail_parts = [f"{k}={v}" for k, v in e.bindings.items()]
            lines.append(f"- **{e.operator}**: {', '.join(detail_parts)}")
        lines.append("")

    EXPORT_PATH.write_text('\n'.join(lines))


def _fmt_bindings(bindings: dict, max_len: int = 120) -> str:
    """Format bindings dict for display."""
    parts = [f"{k}={v}" for k, v in bindings.items()]
    result = ', '.join(parts)
    if len(result) > max_len:
        result = result[:max_len - 3] + '...'
    return result


def main():
    if len(sys.argv) < 2:
        print("Lambda Bridge — Shadow memory system")
        print()
        print("Commands:")
        print("  assert <operator> key=value ...  — store a fact")
        print("  query [operator] [key=value ...] — search facts")
        print("  history [entity]                 — show event history")
        print("  export                           — export to markdown")
        print("  stats                            — show statistics")
        print("  repl                             — interactive mode")
        print()
        print("Recall tracking:")
        print("  recall-score hit|miss|partial    — score the last query")
        print("  recall-stats                     — show recall performance")
        print("  recall-log [N]                   — show last N recall entries")
        print()
        print(f"Store: {STORE_PATH}")
        print(f"Export: {EXPORT_PATH}")
        sys.exit(0)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        'assert': cmd_assert,
        'query': cmd_query,
        'search': cmd_search,
        'history': cmd_history,
        'export': cmd_export,
        'stats': cmd_stats,
        'repl': cmd_repl,
        'recall-score': cmd_recall_score,
        'recall-stats': cmd_recall_stats,
        'recall-log': cmd_recall_log,
    }

    if cmd in commands:
        commands[cmd](args)
    else:
        print(f"Unknown command: {cmd}")
        print("Try: assert, query, history, export, stats, repl")
        sys.exit(1)


if __name__ == '__main__':
    main()
