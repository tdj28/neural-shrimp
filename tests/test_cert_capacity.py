"""Fixtures for tools/cert_capacity.py: finite and unlimited leaves, a tighter parent, unavailable data, exhausted
headroom and a v1 layout, each on a fake /proc and /sys tree."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import cert_capacity as cc  # noqa: E402

GIB = 1024 ** 3
MOUNT2 = "36 28 0:30 / /sys/fs/cgroup rw,nosuid shared:8 - cgroup2 cgroup2 rw,nsdelegate\n"


def tree(tmp, cgroup, mountinfo, files, avail_gib=400):
    (tmp / "proc/self").mkdir(parents=True, exist_ok=True)
    (tmp / "proc/self/cgroup").write_text(cgroup)
    (tmp / "proc/self/mountinfo").write_text(mountinfo)
    (tmp / "proc/meminfo").write_text(f"MemTotal: 527000000 kB\nMemAvailable: {avail_gib * 1024 * 1024} kB\n")
    for rel, text in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp


def test_finite_leaf(tmp_path):
    t = tree(tmp_path, "0::/\n", MOUNT2, {"sys/fs/cgroup/memory.max": str(32 * GIB),
                                          "sys/fs/cgroup/memory.current": str(4 * GIB),
                                          "sys/fs/cgroup/cpu.max": "400000 100000",
                                          "sys/fs/cgroup/cpuset.cpus.effective": "0-7"})
    cap = cc.capacity(t, affinity=8)
    assert cap["unresolved"] == []
    assert cap["effective_memory_bytes"] == 28 * GIB
    assert cap["effective_cpus"] == 4
    assert cc.packing_admission(cap, 1)["admit"]
    assert not cc.packing_admission(cap, 2)["admit"]                 # 2 * 18 + 4 = 40 GiB > 28 GiB


def test_unlimited_leaf_tighter_parent(tmp_path):
    t = tree(tmp_path, "0::/pod/ctr\n", MOUNT2, {"sys/fs/cgroup/pod/ctr/memory.max": "max",
                                                 "sys/fs/cgroup/pod/ctr/memory.current": str(1 * GIB),
                                                 "sys/fs/cgroup/pod/ctr/cpu.max": "max 100000",
                                                 "sys/fs/cgroup/pod/memory.max": str(16 * GIB),
                                                 "sys/fs/cgroup/pod/memory.current": str(6 * GIB),
                                                 "sys/fs/cgroup/pod/cpu.max": "200000 100000",
                                                 "sys/fs/cgroup/cpuset.cpus.effective": "0-95"})
    cap = cc.capacity(t, affinity=96)
    assert cap["unresolved"] == []                                    # the hierarchy root may lack limit files
    assert cap["effective_memory_bytes"] == 10 * GIB                  # the parent's headroom, not the host's
    assert cap["effective_cpus"] == 2                                 # the parent's quota, not 96 host CPUs
    assert not cc.packing_admission(cap, 1)["admit"]                  # 22 GiB > 10 GiB


def test_unlimited_everywhere_visible(tmp_path):
    t = tree(tmp_path, "0::/\n", MOUNT2, {"sys/fs/cgroup/memory.max": "max",
                                          "sys/fs/cgroup/memory.current": str(2 * GIB),
                                          "sys/fs/cgroup/cpu.max": "max 100000",
                                          "sys/fs/cgroup/cpuset.cpus.effective": "0-15"}, avail_gib=100)
    cap = cc.capacity(t, affinity=16)
    assert cap["unresolved"] == []
    assert cap["effective_memory_bytes"] == 100 * GIB                 # host MemAvailable bounds it
    assert cap["effective_cpus"] == 16
    assert cc.packing_admission(cap, 4)["admit"]                      # 76 GiB, 4 CPUs


def test_unavailable_data_refuses(tmp_path):
    t = tree(tmp_path, "0::/\n", MOUNT2, {"sys/fs/cgroup/cpuset.cpus.effective": "0-7"})   # no memory.max, no cpu.max
    cap = cc.capacity(t, affinity=8)
    assert any("memory limit unreadable" in u for u in cap["unresolved"])
    assert any("CPU quota unreadable" in u for u in cap["unresolved"])
    adm = cc.packing_admission(cap, 1)
    assert not adm["admit"] and "unresolved" in adm["reasons"][0]


def test_finite_limit_without_usage_refuses(tmp_path):
    t = tree(tmp_path, "0::/\n", MOUNT2, {"sys/fs/cgroup/memory.max": str(32 * GIB),
                                          "sys/fs/cgroup/cpu.max": "max 100000",
                                          "sys/fs/cgroup/cpuset.cpus.effective": "0-7"})
    cap = cc.capacity(t, affinity=8)
    assert any("without readable usage" in u for u in cap["unresolved"])
    assert not cc.packing_admission(cap, 1)["admit"]


def test_exhausted_headroom(tmp_path):
    t = tree(tmp_path, "0::/\n", MOUNT2, {"sys/fs/cgroup/memory.max": str(8 * GIB),
                                          "sys/fs/cgroup/memory.current": str(9 * GIB),
                                          "sys/fs/cgroup/cpu.max": "max 100000",
                                          "sys/fs/cgroup/cpuset.cpus.effective": "0-7"})
    cap = cc.capacity(t, affinity=8)
    assert cap["effective_memory_bytes"] == 0
    assert not cc.packing_admission(cap, 1)["admit"]


def test_v1_layout(tmp_path):
    mi = ("40 30 0:40 /docker/abc /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
          "41 30 0:41 /docker/abc /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct\n"
          "42 30 0:42 /docker/abc /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset\n")
    cg = "5:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n4:cpuset:/docker/abc\n"
    t = tree(tmp_path, cg, mi, {"sys/fs/cgroup/memory/memory.limit_in_bytes": str(8 * GIB),
                                "sys/fs/cgroup/memory/memory.usage_in_bytes": str(2 * GIB),
                                "sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us": "300000",
                                "sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us": "100000",
                                "sys/fs/cgroup/cpuset/cpuset.effective_cpus": "0-15"})
    cap = cc.capacity(t, affinity=16)
    assert cap["cgroup_version"] == "v1" and cap["unresolved"] == []
    assert cap["effective_memory_bytes"] == 6 * GIB
    assert cap["effective_cpus"] == 3


def test_v1_unlimited_sentinel(tmp_path):
    mi = "40 30 0:40 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n" \
         "41 30 0:41 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu\n" \
         "42 30 0:42 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset\n"
    t = tree(tmp_path, "5:memory:/\n3:cpu:/\n4:cpuset:/\n", mi,
             {"sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712",
              "sys/fs/cgroup/memory/memory.usage_in_bytes": str(GIB),
              "sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1", "sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
              "sys/fs/cgroup/cpuset/cpuset.cpus": "0-31"}, avail_gib=50)
    cap = cc.capacity(t, affinity=32)
    assert cap["unresolved"] == [] and cap["effective_memory_bytes"] == 50 * GIB and cap["effective_cpus"] == 32
