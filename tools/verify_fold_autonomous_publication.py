"""Check FGR3 publication counts, exclusions and figure bytes from its receipt.

Dependency-free so the blog CI can run the same check without research arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prose(path):
    text = path.read_text()
    assert 'FGR3_GROWTH_PUBLICATION_GATE' not in text, 'Growth publication gate is still pending'
    return ' '.join(text.split())


def assets(root, receipt, kind, extension):
    for name, bound in receipt['figures'][kind]['outputs'].items():
        if not name.endswith(extension):
            continue
        if extension == '.pdf' and '-mobile' in name:
            continue
        p = root/name
        assert p.stat().st_size == bound['bytes'] and sha(p) == bound['sha256'], p


def check(blog, short, report=None, journal=None):
    receipt = blog/'fold-return-autonomous.json'
    r = json.loads(receipt.read_text())
    assert r['status'].startswith('Scientific audits complete;')
    assert sha(receipt) == sha(short/'fold-return-autonomous.json')
    v = r['validation']; c = v['one_cycle_route']; train = r['training']
    n = c['evaluated']; tp = c['capture_true_positive']; fp = c['capture_false_positive']; fn = c['capture_false_negative']
    captures = tp+fn; returns = n-captures
    assert c['correct'] == returns-fp+tp
    assert c['always_return_correct'] == returns
    assert c['capture_precision'] == tp/(tp+fp) and c['capture_recall'] == tp/captures
    assert train['rows'] == sum(train['outcomes'].values())
    assert v['rows'] == sum(v['outcomes'].values())
    assert r['input_checks']['validation']['frozen_table_binding_passed']
    assert r['post_test_diagnosis']['relative_reconstruction_error']['maximum'] < 7.5e-14
    windows = [row for w in r['growth']['windows'] if w['start'] is not None
               for row in w['rows'] if row['full_planned_window']]
    assert len(windows) == r['growth']['complete_network_windows']
    assert sum(row['first_half']>0 and row['second_half']>0 for row in windows) == r['growth']['positive_in_both_halves']
    long_text = prose(blog/'index.md'); short_text = prose(short/'index.md')
    for required in (f"sampled {train['rows']:,} departure states",
                     f'identified {tp} captures, missed {fn}, and incorrectly called {fp} returns captures',
                     f"{100*c['capture_precision']:.1f}%", f"{100*c['capture_recall']:.1f}%",
                     'before generating the', 'flagged the interval as unresolved',
                     'later growth', 'first 128 updates', 'not full-window twin exponents'):
        assert required in long_text, required
    assert f'found {tp} of the {captures} captures, missed {fn}, and falsely predicted capture on {fp} returns' in short_text
    for kind in ('returns','growth'):
        assets(blog,r,kind,'.png')
    assets(short,r,'growth','.png')
    for directory in (blog,short):
        extension = next(e for e in json.loads((directory/'provenance.json').read_text())['extensions']
                         if e['id']=='fold-autonomous-return-20261114')
        assert extension['receipts']['fold-return-autonomous.json'] == sha(receipt)
        for item in extension['new_numbers']:
            value = r
            for key in item['field']:
                value = value[key]
            assert value == item['value'], item['id']
            assert item['appears_as'] in prose(directory/'index.md'), item['id']
    if report:
        text=prose(report/'main.tex')
        assert f'found {tp} of the {captures} capture events, missed {fn}' in text
        for kind in ('returns','growth'):assets(report/'figures',r,kind,'.pdf')
    if journal:
        text=prose(journal/'main.tex')
        assert f'found {tp} of {captures} period-3 approaches, missed {fn}' in text
        assets(journal/'figures',r,'growth','.pdf')
    print('FGR3 publication: audited counts, rare-event scores, exclusions, growth gates and asset hashes verified for both blog posts.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blog',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--short',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--journal',type=Path)
    a=p.parse_args()
    check(a.blog,a.short or a.blog.parent/'language-model-feedback-short-read',a.report,a.journal)
