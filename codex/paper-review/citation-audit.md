# Task13 bibliography verification — 13 October 2026

Pinned bibliography: `paper/llm/references.bib`, SHA256 `4a062ece54122fb866387dba2dff13b132426792388520a21bd4f39d36eb4c1f`, at commit1491f37972e706426d3343ce5b69fa91ab89da01. The actual manuscript cites63 distinct keys; the bibliography contains68 entries. No cited key is absent. Five entries are presently uncited. The33-key independent root review is preserved byte-for-byte in `root-citation-audit.md/.json`, with its Crossref records. This note supplies the disjoint30 cited entries and five uncited entries, completing record verification for all68.

“Matches” means the cited title, listed authors/order, identifier and year identify the stated primary version; it does not mean every scientific claim in that source was independently re-derived. Preprints do not have journal volume/page numbers. Their arXiv identifier is verified; an arXiv DOI identifies the preprint, not journal peer review. Retaining an explicitly identified preprint is valid even when a later conference version exists. Do not mix differing author/version records.

## Required correction and optional publication updates

- **Required:** `facanha2013endoskeletons`: **Wilson Façanha**, not Wellington. Publisher-deposited DOI10.1016/j.physleta.2013.03.025; all other supplied bibliographic fields match. Root's independent report gives the primary link and complete record.
- **Scientific attribution correction:** manuscript688–690 misdescribes TAPS's0.8 result and overstates absence of an explanation; see C07 of `consistency-review.md`. This is a claim-to-source error even though the citation metadata are correct.
- **Optional published-version upgrades:** PonderLM→ICLR2026; Coconut→COLM2025; Geiping→NeurIPS38:46242–46293(2025), DOI10.52202/085713-1380; Tomihari→NeurIPS38:155236–155269(2025), DOI10.52202/085713-4675; STARS→ICML/PMLR306:144670–144682(2026); Bai2021→ICML/PMLR139:554–565. Exact candidate entries are provided separately, not applied to the paper. Current arXiv citations are not mislabeled journal articles.
- Other optional root completeness: Ashok–May pages69806–69837, Gallas issue18, Stenzinger issue2.
- Several earlier `@misc` entries omit URLs/DOIs/version notes. The table below supplies an explicit checked version to add. Preserve first-submission year when citing that preprint, and disclose the checked later version rather than silently changing scientific scope.

## Thirty cited entries checked in this lane

All links below are primary arXiv, publisher/proceedings or official conference records, accessed13October2026. All current title/author/identifier fields match, subject to the version choices and publication updates shown.

|Key|Verified primary version/record|Disposition|
|---|---|---|
|`zeng2025ponderlm`|[arXiv2505.20674](https://arxiv.org/abs/2505.20674), first27May2025, v3 20Feb2026; [ICLR2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94572a66c6b0710c5263f15d8ef6b463-Abstract-Conference.html), nine authors beginning Boyi Zeng, Shixiang Song, Siyuan Huang.|Matches preprint; optional2026 conference replacement. No DOI/pages supplied on the inspected ICLR record.|
|`hao2024coconut`|[arXiv2412.06769](https://arxiv.org/abs/2412.06769), first9Dec2024, v4 23Aug2026; [official COLM2025 paper](https://openreview.net/pdf?id=Itxz7S4Ip3), seven authors, title and published-as-COLM header verified.|Matches preprint; optional COLM2025 replacement. OpenReview forum later challenged the browser, but the official PDF was accessible; no invented pages/DOI.|
|`li2025quasilyapunov`|[arXiv2503.13530v1](https://arxiv.org/abs/2503.13530v1),15Mar2025; Xiaojian Li, Yongkang Leng, Ruiqing Ding, Hangjie Mo, Shanlin Yang.|Matches; add URL/version if desired.|
|`jaca2026chaos`|[arXiv2607.27805v1](https://arxiv.org/abs/2607.27805v1),30Jul2026; Gregorio Jaca, Kristóf Benedek, János Török.|Matches abstract-record author order; do not silently replace it with the differently ordered HTML byline.|
|`arnold2024phase`|[arXiv2405.17088v1](https://arxiv.org/abs/2405.17088v1),27May2024; Julian Arnold, Flemming Holtorf, Frank Schäfer, Niels Lörch.|Matches. A later review submission is not publication authority.|
|`nakaishi2024phase`|[arXiv2406.05335](https://arxiv.org/abs/2406.05335), first8Jun2024, currentv3 6Jun2026; Kai Nakaishi, Yoshihiko Nishikawa, Koji Hukushima.|Matches; identify the chosen version.|
|`sun2025phase`|[arXiv2501.16241v1](https://arxiv.org/abs/2501.16241v1),27Jan2025; Youran Sun, Babak Haghighat.|Matches.|
|`ruan2026criticality`|[arXiv2606.06238v1](https://arxiv.org/abs/2606.06238v1),4Jun2026; Huajian Ruan, Jinyang Li, Xingyu Guo, Lingxiao Wang.|Matches; the comment aboutPAI2026 is not used to invent a proceedings citation.|
|`johnson2026temperature`|[arXiv2608.00939v1](https://arxiv.org/abs/2608.00939v1),2Aug2026; Neil F. Johnson, Frank Yingjie Huo, Bella Xinrui Li.|Matches.|
|`wang2025attractor`|[arXiv2502.15208](https://arxiv.org/abs/2502.15208), first21Feb2025, v2 16May2025; Zhilin Wang, Yafu Li, Jianhao Yan, Yu Cheng, Yue Zhang.|Matches.|
|`tomihari2025recurrent`|[arXiv2505.19458](https://arxiv.org/abs/2505.19458), v4 5Nov2025; [NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cca569b38ec3612904e625dad7a743dc-Abstract-Conference.html), Akiyoshi Tomihari and Ryo Karakida; DOI10.52202/085713-4675,38:155236–155269(2025).|Matches; publication metadata also checked with publisher-deposited Crossref, saved separately.|
|`geiping2025recurrent`|[arXiv2502.05171](https://arxiv.org/abs/2502.05171), v2 17Feb2025; [NeurIPS record](https://proceedings.nips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html), nine authors; DOI10.52202/085713-1380,38:46242–46293(2025).|Matches; optional published version. Its conference record uses Brian Bartoldson; the preprint uses Brian R. Bartoldson. Candidate follows conference metadata.|
|`zhu2025ouro`|[arXiv2510.25741](https://arxiv.org/abs/2510.25741), first29Oct2025, v5 1Jul2026; authors begin Rui-Jie Zhu, Zixuan Wang, Kai Hua.|Matches listed authors/title; identify the chosen version rather than interpreting the2026 revision as a2025 venue.|
|`yang2026stars`|[arXiv2605.26733v1](https://arxiv.org/abs/2605.26733v1),26May2026; [PMLR306](https://proceedings.mlr.press/v306/yang26t.html),144670–144682, ICML2026; seven authors Xiao-Wen Yang through Yu-Feng Li.|Matches; complete publication entry available. PMLR page supplies no DOI.|
|`luo2026prediction`|[arXiv2609.21383v1](https://arxiv.org/abs/2609.21383v1),18Sep2026; Xinyue Luo, Fei Yu.|Matches.|
|`ko2026attractor`|[arXiv2606.30571v1](https://arxiv.org/abs/2606.30571v1),29Jun2026; Ting-Wen Ko, Jonas Geiping.|Matches.|
|`bai2019deq`|[arXiv1909.01377](https://arxiv.org/abs/1909.01377), first3Sep2019, v2 28Oct2019; [NeurIPS32](https://proceedings.neurips.cc/paper_files/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html); Shaojie Bai, J. Zico Kolter, Vladlen Koltun.|Matches. Official landing confirms volume32 but no page range/DOI was exposed; its BibTeX link failed. No unsupported pagination proposed.|
|`bai2021stabilizing`|[arXiv2106.14342v1](https://arxiv.org/abs/2106.14342v1),28Jun2021; [PMLR139:554–565](https://proceedings.mlr.press/v139/bai21b.html), ICML2021; Shaojie Bai, Vladlen Koltun, Zico Kolter.|Matches; publication author/order/title/pages/year verified.|
|`holtzman2019degeneration`|[arXiv1904.09751](https://arxiv.org/abs/1904.09751), first22Apr2019, v2 14Feb2020; Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi.|Matches preprint. ICLR2020 is indicated in primary arXiv metadata; direct OpenReview access hit a challenge, so retain the arXiv entry rather than pretend the venue record was read.|
|`xu2022loop`|[arXiv2206.02369](https://arxiv.org/abs/2206.02369), first6Jun2022, v2 9Oct2022; [NeurIPS35](https://proceedings.neurips.cc/paper_files/paper/2022/hash/148c0aeea1c5da82f4fa86a09d4190da-Abstract-Conference.html); Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, Jian Li.|Matches. Landing confirms venue/year/volume; BibTeX retrieval failed, so page range/DOI not asserted.|
|`kim2026phase`|[arXiv2608.26556v1](https://arxiv.org/abs/2608.26556v1),27Aug2026, Gunn Kim; DOI10.48550/arXiv.2608.26556.|Matches explicit version. Earlier Task3 full-text review supports the deliberately scoped prior-work discussion.|
|`zhang2025softthinking`|[arXiv2505.15778v1](https://arxiv.org/abs/2505.15778v1),21May2025, eight authors as currently listed.|Matches explicit arXiv record. Prior independently checked NeurIPS landing has a different six-author roster; do not attach that venue to this roster without choosing the exact version.|
|`wu2025singlethreaded`|[arXiv2508.03440v4](https://arxiv.org/abs/2508.03440v4),16Oct2025, seven authors Junhong Wu through Hua Wu.|Matches explicit version/title; earlier versions had a different title.|
|`prairie2026parcae`|[arXiv2604.12946v1](https://arxiv.org/abs/2604.12946v1),14Apr2026; Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, Daniel Y. Fu.|Matches explicit version.|
|`buehlmaier2026layernorm`|[arXiv2607.10681v1](https://arxiv.org/abs/2607.10681v1),12Jul2026; Matthias M. M. Buehlmaier.|Matches explicit version.|
|`labovich2026stability`|[arXiv2604.15259v2](https://arxiv.org/abs/2604.15259v2),22Apr2026; Asher Labovich.|Matches explicit version; unversioned retrieval failed once, versioned primary record succeeded.|
|`movahedi2026fixedpoint`|[arXiv2606.18206v1](https://arxiv.org/abs/2606.18206v1),16Jun2026; eight authors Sajad Movahedi through Antonio Orvieto.|Matches full supplied author list and explicit version.|
|`zou2023representation`|[arXiv2310.01405v4](https://arxiv.org/abs/2310.01405v4),3Mar2025, first2Oct2023;21 authors Andy Zou through Dan Hendrycks.|Matches existing explicit first-year/version convention.|
|`liu2026learningchaos`|[arXiv2609.38814v1](https://arxiv.org/abs/2609.38814v1),30Sep2026; eight authors Yilun Liu through Yunpu Ma.|Matches. Task3 full-text scope is a transformer trained to model logistic-map data, not a frozen pretrained language-model mechanism.|
|`wang2026taps`|[arXiv2609.36653v1](https://arxiv.org/abs/2609.36653v1),29Sep2026; Boyuan Wang, Chengyao Yu, Jiaxi Ren, Hongxin Wei, Bingyi Jing, Yuxin Tao.|Metadata matches; manuscript claim correctionC07 is required.|

## Five uncited entries, also verified

|Key|Primary record|Disposition|
|---|---|---|
|`clark2026lyapunov`|[arXiv2610.12426v1](https://arxiv.org/abs/2610.12426v1),8Oct2026; David G. Clark; Lyapunov spectrum of random neural networks.|Matches; currently not cited in this LLM draft. No implication that the NN-training paper's citation requirement is satisfied here.|
|`liu2025truthdecay`|[arXiv2503.11656v1](https://arxiv.org/abs/2503.11656v1); eight authors Joshua Liu through Vasu Sharma.|Matches title/identifier/year. The primary page displays4Feb2025 despite the2503 identifier; the current entry prudently specifies v1 without inventing a date.|
|`tan2025persuasion`|[ACL Anthology2025.emnlp-main.81](https://aclanthology.org/2025.emnlp-main.81/), DOI10.18653/v1/2025.emnlp-main.81, EMNLP2025:1550–1575; five authors Bryan Chen Zhengyu Tan through Roy Ka-Wei Lee.|All supplied publication metadata match.|
|`huang2026loopedfixed`|[arXiv2610.06833v1](https://arxiv.org/abs/2610.06833v1),5Oct2026; seven authors Benhao Huang through Xuezhe Ma.|Matches.|
|`wang2026jacobianlens`|[arXiv2609.01924v1](https://arxiv.org/abs/2609.01924v1),1Sep2026; Wenlong Wang, Fergal Reid.|Matches.|

## Search and verification boundary

Discovery used exact titles/identifiers and official-site queries, including `site.openreview.net PonderLM`, `site.proceedings.neurips.cc Recurrent Self-Attention Dynamics`, `site.proceedings.mlr.press Stabilizing Recurrent Dynamics`, and exact-title searches for Coconut, Geiping, Deep Equilibrium Models, Holtzman and Xu. Claims/metadata rely on the linked primary records, not secondary search summaries. Direct arXiv records supplied all35 identifier/title/author checks; the conference pages/PMLR and publisher-deposited Crossref supplied the optional publication fields. TAPS Table3 and its theoretical sections were read for the substantive citation correction. No paywall was purchased, no authors contacted, and no search result was treated as proof of novelty.

The bibliography is therefore resolvable and mostly accurate; one author correction is required. Publication upgrades are optional choices, with precise unverified page/DOI gaps disclosed above. The strongest contribution should be stated positively (specified pretrained-map periodic orbits, continued boundaries, derivative passage predictions and scoped behavioral experiments), not as an unrestricted first-ever neural bifurcation or warning result.
