"""Manufactured controls, run once with the admitted tokenizer profile only."""
import traceback


class Fake:
    def __init__(self, text, good=True):
        self.text, self.good = text, good
    def __call__(self, text, **kwargs):
        return {'input_ids': [1]}
    def decode(self, ids, **kwargs):
        return self.text if ids else ''


def run(api, extract):
    results = []
    def check(name, f):
        try:
            f()
            results.append({'name': name, 'pass': True})
        except Exception as e:
            results.append({'name': name, 'pass': False, 'error': repr(e), 'traceback': traceback.format_exc()})
    def eq(a, b):
        assert a == b, (a, b)
    allowed = ({1, 2}, {1, 2})
    left, right = Fake('\\boxed{1}'), Fake('\\boxed{2}')
    def one(row):
        return api.decode_row(row, left, right, allowed, extract)
    def batch(rows):
        return api.condition('fixture', rows, set(range(64)), list(range(64)), left, right, allowed, extract)
    check('independent_box_and_fallback', lambda: eq([extract('x \\boxed{-2.0}', 'pm2'), extract('x 4 y 7', 'pm2')], ['-2', '7']))
    check('valid_ids_change_answer', lambda: eq((one({'answer_ids': [1]})['answers_changed'], one({'answer_ids': [1]})['qualified_ids']), (True, True)))
    check('missing_ids_literal_empty_unqualified', lambda: eq((one({})['answers_changed'], one({})['qualified_ids'], one({})['literal_missing_ids_projection']), (False, False, True)))
    check('empty_ids_valid', lambda: eq((one({'answer_ids': []})['answers_changed'], one({'answer_ids': []})['qualified_ids']), (False, True)))
    def invalid():
        for ids in (None, '1', [-1], [True], [3]):
            r = one({'answer_ids': ids})
            assert r['input_errors'] and r['answers_changed'] is None
    check('invalid_and_unmapped_ids_retained', invalid)
    rows = [{'index': i, 'answer_ids': [1]} for i in range(64)]
    check('complete64', lambda: eq((batch(rows)[0]['full_64_qualified'], batch(rows)[0]['answers_changed_by_decoder']), (True, 64)))
    def coverage():
        report, ledger = batch(rows[:-1] + [rows[0], {'index': 999, 'answer_ids': []}])
        assert report['missing_indices'] == [63] and report['duplicate_lines'] == [64]
        assert report['extra_or_invalid_index_lines'] == [65]
        assert ledger[63]['duplicate_of_lines'] == [1] and not ledger[0]['selected_last_write']
        assert not report['strict_cohort_complete']
    check('missing_extra_duplicate_last_write', coverage)
    def fallback():
        good = Fake(api.ROUNDTRIP)
        chosen, trace = api.checked_choice(Fake('bad'), lambda: good)
        assert chosen is good and trace['fallback_used']
        chosen, trace = api.checked_choice(good, lambda: (_ for _ in ()).throw(AssertionError('must not load fallback')))
        assert chosen is good and not trace['fallback_used']
    check('literal_checked_choice', fallback)
    def refusal():
        try:
            api.checked_choice(Fake('bad'), lambda: Fake('also bad'))
        except ValueError:
            return
        raise AssertionError('missing checked-loader refusal')
    check('failed_fallback_refusal', refusal)
    return {'groups': len(results), 'all_pass': all(x['pass'] for x in results), 'results': results}
