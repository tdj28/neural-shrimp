"""Literal Task11 arm membership; no discovery from scored outcomes."""
CAPTURE = ('p1-free','p1-h040','p1-h050','p1-h050s2','p1-h060','p1-h060s2','p1-h070',
           'p1-r1-060','p1-r1-080','p1-r2-080','p1-ck-060','p1-ck-080')
HISTORY = ('p1-ramp-s1','p1-ramp-s2','p1-c040-s2','p1-exit030','p1-exit020','p1-exit000',
           'p1-exitm020','p1-c030-s1','p1-c030-s2','p1-exit040-s1','p1-exit040-s2',
           'p1-exit030-s2','p1-exit000-s2','p1-exitm020-s2')
L6 = ('p1-ctl-none','p1-ctl-trig-cancel','p1-ctl-trig-over','p1-ctl-trig-long',
      'p1-ctl-trig-kind','p1-ctl-const-02','p1-ctl-const-04')
L10 = ('p1-ctl2-none','p1-ctl2-const-02','p1-ctl2-const-04','p1-ctl2-regen-06x3',
       'p1-ctl2-regen-04x0','p1-ctl2-regen-04x3','p1-ctl2-trig-06x3')
TRAIN = ('p1-det040','p1-det045')
TEST = ('p1-conf040','p1-conf045')
ALL = CAPTURE+HISTORY+L6+L10+TRAIN+TEST
assert len(ALL)==44 and len(set(ALL))==44
POOLS = {'ramp':('p1-ramp-s1','p1-ramp-s2'), 'fresh040':('p1-h040','p1-c040-s2'),
         'fresh050':('p1-h050','p1-h050s2'), 'fresh060':('p1-h060','p1-h060s2'),
         'fresh030':('p1-c030-s1','p1-c030-s2'), 'fresh000':('p1-free',),
         'hold040':('p1-exit040-s1','p1-exit040-s2'), 'hold030':('p1-exit030','p1-exit030-s2'),
         'hold020':('p1-exit020',), 'hold000':('p1-exit000','p1-exit000-s2'),
         'holdm020':('p1-exitm020','p1-exitm020-s2')}
CONTRASTS = (('ramp','fresh040'),('ramp','fresh050'),('hold040','fresh040'),('hold030','fresh030'))
SECOND = 's-nlp/roberta_toxicity_classifier'
SECOND_REVISION = '048c25bb1e199b98802784f96325f4840f22145d'
FIRST = 'unitary/toxic-bert'
FIRST_REVISION = '4d6c22e74ba2fdd26bc4f7238f50766b045a0d94'


def replicas(name):
    if name not in ALL: raise ValueError('unknown arm')
    return 256 if name in TRAIN else 512 if name in TEST else 128


def filenames(name):
    if name not in ALL: raise ValueError('unknown arm')
    names=['sample.json','hostility.npz','hostility.json','hostility-second.npz']
    if name in L6+L10: names+=['turns.npz']
    if name in TRAIN+TEST: names+=['in-context.npz','activations.npy']
    return names
