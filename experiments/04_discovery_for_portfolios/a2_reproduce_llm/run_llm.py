"""
Run the Sokolov et al. LLM pipeline on one graph: C1 x10 (fresh chats, edges unioned), C2 (directions),
then C3 repeated in the same chat until the LLM says the DAG is valid (max 5 rounds).
Backends: ollama (automatic, local gpt-oss) or manual (Claude via clipboard, pasted into claude.ai).
Saves every prompt, reply and the final DAG under a2_reproduce_llm/responses/<graph>/<label>/run<NN>/; resumable.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / 'prompts'))
import sokolov_prompts as P
from common.paths import GRAPHS, RESPONSES

N_EXPLORE = 10
MAX_VALIDATE = 5


class Ollama:
    def __init__(self, model, host='http://127.0.0.1:11434'):
        import ollama
        # explicit host: an OLLAMA_HOST of 0.0.0.0 works for the server but not as a client address on Windows
        self.client, self.model, self.label = ollama.Client(host=host), model, model.replace(':', '-')

    def ask(self, messages, step, fresh):
        reply = self.client.chat(model=self.model, messages=messages)['message']
        return reply['content'], reply.get('thinking') or ''


class Manual:
    def __init__(self, label):
        import pyperclip
        self.clip, self.label, self.last_reply = pyperclip, label, None

    @staticmethod
    def flush_keyboard():
        """Drop anything typed or pasted into the terminal, so stray lines can't count as Enter presses."""
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        except ImportError:
            pass

    def ask(self, messages, step, fresh):
        prompt = messages[-1]['content']
        self.clip.copy(prompt)
        where = "a NEW chat (outside any project)" if fresh else "the SAME chat as C2"
        print(f"\n[{step}] Prompt copied. Paste it into {where} and send.")
        while True:
            self.flush_keyboard()
            input("    When Claude finishes, click its Copy button, then press Enter here (don't paste here)... ")
            reply = self.clip.paste()
            if not reply.strip() or reply == prompt:
                print("    Clipboard still holds the prompt (or is empty). Copy Claude's reply and try again.")
            elif reply == self.last_reply:
                print("    Clipboard holds the PREVIOUS step's reply. Copy this step's reply and try again.")
            else:
                self.last_reply = reply
                return reply, ''


def parse_json(text):
    """JSON from a ```json block, else from the first '{' to the last '}'; None if neither parses."""
    for candidate in re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S) + [text[text.find('{'):text.rfind('}') + 1]]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def step(backend, messages, name, run_dir, fresh, retries=2):
    """One LLM call with on-disk caching: reuse saved reply if present, else ask (retrying unparseable replies)."""
    reply_file = run_dir / f"{name}_reply.txt"
    (run_dir / f"{name}_prompt.txt").write_text(messages[-1]['content'], encoding='utf-8')
    if reply_file.exists():
        reply = reply_file.read_text(encoding='utf-8')
        return reply, parse_json(reply)
    for attempt in range(retries + 1):
        t0 = time.time()
        reply, thinking = backend.ask(messages, name, fresh)
        parsed = parse_json(reply)
        if parsed is not None:
            break
        print(f"    {name}: reply was not valid JSON (attempt {attempt + 1})")
    reply_file.write_text(reply, encoding='utf-8')
    if thinking:
        (run_dir / f"{name}_thinking.txt").write_text(thinking, encoding='utf-8')
    print(f"    {name}: {time.time() - t0:.0f}s, parsed={parsed is not None}")
    return reply, parsed


def edges_from(parsed, names, run_dir, name):
    """(source, target, justification) tuples, keeping only known feature names; logs anything dropped."""
    edges, dropped = [], []
    for e in (parsed or {}).get('edges', []):
        s, t = str(e.get('source', '')).strip(), str(e.get('target', '')).strip()
        (edges if s in names and t in names and s != t else dropped).append((s, t, e.get('justification', '')))
    if dropped:
        (run_dir / f"{name}_dropped.json").write_text(json.dumps(dropped, indent=1), encoding='utf-8')
    return edges


def fmt_definitions(variables):
    return '\n'.join(f"Feature Name: {v['name']}, Feature Definition: {v['definition']}" for v in variables)


def apply_corrections(dag, corrections, names):
    """dag: {(s, t): justification}. missing -> add, nonexistent -> remove, direction -> flip."""
    for c in corrections:
        s, t, why = str(c.get('source', '')).strip(), str(c.get('target', '')).strip(), c.get('justification', '')
        if s not in names or t not in names or s == t:
            continue
        kind = c.get('type')
        if kind == 'missing' and (s, t) not in dag and (t, s) not in dag:
            dag[(s, t)] = why
        elif kind == 'nonexistent':
            dag.pop((s, t), None) or dag.pop((t, s), None)
        elif kind == 'direction' and (s, t) in dag:
            dag.pop((s, t))
            dag[(t, s)] = why
    return dag


def run(graph, backend, run_no):
    spec = json.loads((GRAPHS / f"{graph}_variables.json").read_text(encoding='utf-8'))
    variables = spec['variables']
    names = {v['name'] for v in variables}
    name_list, defs = ', '.join(v['name'] for v in variables), fmt_definitions(variables)
    run_dir = RESPONSES / graph / backend.label / f"run{run_no:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    head = P.SYSTEM + '\n\n'     # system text sent inside the user message for every backend (claude.ai has no system field)

    # C1: exploration, 10 independent calls, union of undirected edges
    c1_prompt = head + P.C1.format(names=name_list, definitions=defs) + P.JSON_EDGES
    undirected, failed = set(), []
    for i in range(1, N_EXPLORE + 1):
        _, parsed = step(backend, [{'role': 'user', 'content': c1_prompt}], f"c1_{i:02d}", run_dir, fresh=True)
        failed += [f"c1_{i:02d}"] if parsed is None else []
        undirected |= {frozenset((s, t)) for s, t, _ in edges_from(parsed, names, run_dir, f"c1_{i:02d}")}
    pairs = sorted(tuple(sorted(p)) for p in undirected)
    print(f"C1: {len(pairs)} undirected edges after {N_EXPLORE} runs")

    # C2: inference, directions for every edge (starts the shared chat)
    c2_prompt = head + P.C2.format(edges=''.join(f"({a}, {b})" for a, b in pairs), names=name_list,
                                   definitions=defs) + P.JSON_EDGES
    chat = [{'role': 'user', 'content': c2_prompt}]
    c2_reply, parsed = step(backend, chat, 'c2', run_dir, fresh=True)
    failed += ['c2'] if parsed is None else []
    chat.append({'role': 'assistant', 'content': c2_reply})
    dag = {(s, t): why for s, t, why in edges_from(parsed, names, run_dir, 'c2')}

    # C3: validation loop in the same chat
    rounds, valid = 0, False
    while rounds < MAX_VALIDATE and not valid:
        rounds += 1
        dag_text = '\n'.join(f"({s}, {t}, {why})" for (s, t), why in dag.items())
        chat.append({'role': 'user', 'content': P.C3.format(dag=dag_text) + P.JSON_VALIDATE})
        c3_reply, parsed = step(backend, chat, f"c3_{rounds:02d}", run_dir, fresh=False)
        failed += [f"c3_{rounds:02d}"] if parsed is None else []
        chat.append({'role': 'assistant', 'content': c3_reply})
        corrections = (parsed or {}).get('corrections', [])
        valid = bool((parsed or {}).get('valid')) and not corrections
        dag = apply_corrections(dag, corrections, names)
        print(f"C3 round {rounds}: valid={valid}, {len(corrections)} corrections, {len(dag)} edges")

    is_dag = nx.is_directed_acyclic_graph(nx.DiGraph(list(dag)))
    result = {'graph': graph, 'backend': backend.label, 'run': run_no, 'date': datetime.now().isoformat(timespec='seconds'),
              'n_undirected_c1': len(pairs), 'validation_rounds': rounds, 'validated': valid, 'is_dag': is_dag, 'failed_calls': failed,
              'edges': [{'source': s, 'target': t, 'justification': why} for (s, t), why in dag.items()]}
    (run_dir / 'final_dag.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(f"final: {len(dag)} edges, acyclic={is_dag}, validated={valid}, failed calls={failed or 'none'}")
    return result


def check(graph, result):
    """Gate check only: compare with the answer key, read after the run (never shown to the LLM)."""
    truth = {tuple(e) for e in json.loads((GRAPHS / f"{graph}_truth.json").read_text())['edges']}
    found = {(e['source'], e['target']) for e in result['edges']}
    skeleton = lambda es: {frozenset(e) for e in es}
    print(f"vs truth: {len(found & truth)}/{len(truth)} edges right incl. direction, "
          f"{len(skeleton(found) & skeleton(truth))}/{len(truth)} right ignoring direction, "
          f"{len(skeleton(found) - skeleton(truth))} extra")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph')
    parser.add_argument('--backend', choices=['ollama', 'manual'], default='ollama')
    parser.add_argument('--model', default='gpt-oss-exp04', help="ollama model, or a label for manual runs")
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--check', action='store_true', help="compare with the answer key afterwards")
    args = parser.parse_args()
    backend = Ollama(args.model) if args.backend == 'ollama' else Manual(args.model)
    result = run(args.graph, backend, args.run)
    if args.check:
        check(args.graph, result)