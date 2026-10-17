#!/usr/bin/env python3
"""Focused manufactured TeX delimiter checks; no target input or indexing."""
import argparse
import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('check.py')
    spec = importlib.util.spec_from_file_location('checker_delimiter_controls', source)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    cases = [
        ('row_break_bracket', r'\\[2pt] period-8 $x-2$', ['2', '8', '-2'], ['$x-2$']),
        ('row_break_then_dollar', r'\\$-8$ OLMo-2', ['-8', '2'], ['$-8$']),
        ('adjacent_inline', r'$-1$$-2$ period-8', ['-1', '-2', '8'], ['$-1$', '$-2$']),
        ('escaped_then_inline', r'\$$-8$ period-3', ['-8', '3'], ['$-8$']),
        ('row_break_then_display', r'\\\[x-2\] period-8', ['-2', '8'], [r'\[x-2\]']),
        ('display_then_inline', r'$$x-3$$ period-6 $-2$', ['-3', '6', '-2'], ['$$x-3$$', '$-2$']),
        ('masked_figure_fragment', r'\includegraphics[width=0.49\linewidth]{a.png}\\[2pt]\caption{Top: period-8 ($T=-0.135$).}', ['2', '8', '-0.135'], ['$T=-0.135$']),
    ]
    rows = []
    for name, original, expected, expected_spans in cases:
        text = checker.clean_tex(original)[0] if name == 'masked_figure_fragment' else original
        tokens = list(checker.numeric_tokens(text))
        actual = [str(t[2]) for t in tokens]
        spans = [text[a:b] for a, b in checker.math_spans(text)]
        passed = ([Decimal(x) for x in actual] == [Decimal(x) for x in expected]
                  and spans == expected_spans)
        rows.append(dict(name=name, source=original, expected=expected, actual=actual,
                         expected_math_spans=expected_spans, actual_math_spans=spans,
                         passed=passed))
    report = dict(scope='Seven focused manufactured delimiter groups only; no target files, extraction sweep, old controls or profile repeated.',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  groups=rows, passed=sum(r['passed'] for r in rows),
                  failed=sum(not r['passed'] for r in rows))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
