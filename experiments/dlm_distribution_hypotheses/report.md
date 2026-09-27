# 저장된 분포와 실제 mask에서 유도한 가설

## Objective

통계를 충분히 관찰하고, 관측→해석→가설→반증 조건의 연결을 만든다. 먼저 DLM 기능을 상상한 뒤 통계를 끼워 맞추거나, log-variance를 새 방법으로 포장하지 않는다.

## Actual setup

- LLaDA-8B-Base revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`.
- 기존 224 projections의 원본 weight/score 통계, 같은 8문장×10 timestep=80 corrupted states 및 8 clean inputs 재사용. 입력 manifest hash 및 state 순서 확인.
- Weight/score moments·5×mean tail count/energy는 전체 tensor의 정확한 통계. 표에 쓰인 score median/quantiles는 projection당 131,072개 표본 추정치. Raw activation은 state당 1,024개 scalar 표본이며 전체 activation histogram으로 부르지 않는다.
- 저장된 모든 channel energy와 기존 native batch1 module-only50 signed CE, 12쌍의 fixed-budget exchange(16 validation articles×MC128), 11개 기존 allocation 결과를 연결했다.
- 추가 CPU pass에서 6개 checkpoint shard hash를 재확인하고, 기존 Uniform rowwise/layer-global mask 각각224개를 읽어 count·tail retention·removed diagonal energy·row sparsity를 정확히 집계했다. 총202.62초. 새 GPU, forward, pruning mask, generation, downstream 평가 없음.
- 모든 quartile 표는 해당 type의 8개 projection 통계의 산술평균이다. 비율의 평균과 pooled ratio를 구분한다. Q/K/V 입력 통계는 실제로 동일하고, ff_proj/up_proj 입력 통계도 동일함을80개 state 전체에서 확인했다. 이들을 독립적인 activation 관측으로 세지 않는다.

## Results 1 — 후반의 증가는 극단값만의 증가가 아니다

초기=block0–7, 후반=block24–31. 아래 배율은 후반 평균/초기 평균이다.

| Type | 평균 score 배율 | 중앙값 배율 | 99백분위 배율 | 5×mean 초과 score의 제곱합 비중: 초기→후반 |
|---|---:|---:|---:|---:|
| Q | 4.62× | 7.20× | 2.37× | 94.37%→37.83% |
| K | 4.48× | 7.35× | 2.26× | 97.82%→62.13% |
| V | 11.12× | 14.70× | 6.31× | 57.35%→4.50% |
| Attention output | 16.82× | 23.91× | 9.70× | 52.46%→6.14% |
| FF projection | 5.39× | 5.52× | 4.99× | 25.91%→22.85% |
| Up projection | 6.15× | 6.30× | 5.73× | 17.39%→10.55% |
| FF output | 16.89× | 16.15× | 18.51× | 42.93%→35.84% |

특히Q/K/V/attention outputでは中央部の増加が99百分位の増加より大きい。Qのmedian/meanは.433→.795、Vは.563→.832。初期の尖った分布が単に拡大したという説明では足りない。

ただし全type共通の単調な平坦化ではない。FF-outのtail energyはquartile順42.93%,20.59%,12.94%,35.84%と後半で戻り、block31単独では78.59%。MLP入力側もQ/K/Vとは異なる。

Weight自体の平均絶対値はQ/Kで後半に0.877×/0.882×へ低下する。それでもscoreが4.5倍前後へ増えるため、「後半のweightが大きいから」だけでは説明できない。既存backtraceのactivation-scale寄与と整合する。座標エネルギーの集中度は、共分散行列のrankや機能的冗長性そのものではない。

## Results 2 — 大きな尾部は、良いmaskでも悪いmaskでもすでに残っている

固定した共通基準として、各projectionの **dense Wanda score > 5×そのprojectionのdense平均** をtailと定義した。これはOWL block共通thresholdと同一ではない。

| 実際の既存50% mask | Tail総数 | Tail除去数 | Full validation NELBO | PPL bound推定値 |
|---|---:|---:|---:|---:|
| Uniform rowwise | 38,963,987 | **0** | 2.550805 | 12.817419 |
| Uniform layer-global | 38,963,987 | **0** | 2.663777 | 14.350387 |

両方とも224/224 matricesで上記tailを完全に保存する。総pruned数も両方3,489,660,928。従って、この二つの既存モデルの性能差は **このtail集合が消えたかどうか** では説明できない。全てのoutlier定義やoutlier機能が無意味という結論ではない。

さらに、既存12交換に使った24 matrices・各2方向の境界集合（各786,432 weights、合計37,748,736個の異なる座標）には、このtailに入るweightが一つもない。交換境界の平均scoreは、それぞれのdense全体平均の0.119–0.884倍だった。

同じ50%でも除去するdense score²の比率は深さで異なる：

| Type | Rowwise50で除去したscore² / 全score²：初期→後半 |
|---|---:|
| Q | 0.224%→4.401% |
| K | 0.076%→2.670% |
| V | 1.670%→6.715% |
| Attention output | 1.532%→5.748% |
| FF output | 2.796%→3.206% |

これはdiagonal energyであり実際のreconstruction errorでもNELBO損傷でもない。それでも「同じ除去率が同じ分布領域・同じエネルギー割合を除く」という解釈には具体的な反例になる。

## Results 3 — Layer-globalは低scoreの出力側に除去を集中させた

初期8blockのmatrix平均sparsityは、layer-globalでattention output71.99%、FF-out77.42%、FF-proj30.18%、up30.79%。同じlayer50%のまま内部の負担は大きく偏る。

FF-outのdense score²除去比率は初期でrowwise2.80%→layer-global13.25%、block8–15で4.78%→22.12%。Block8–15のFF-outでは行の91.62%が75%超のsparsityとなる。これらでも前述の5×mean tailは完全保存される。

したがって「極端な大scoreを残す」ことと「出力側projectionに十分な非tail接続を残す」ことは、実際のmaskで分離している。ただし、この再配分が性能差の全てを引き起こしたと帰属するには、layer budget・row分布・sparse-prefix影響を分離した介入が必要。

## Results 4 — 同じactivationでもweightとの対応が大きく異なる

入力channel energyをA_j、weightをW_ijとして、次を記述統計として算出した：

`R = mean_ij(W_ij² A_j) / [mean_ij(W_ij²) × mean_j(A_j)]`

列ごとのweight energyが入力energyと無関係なら1になる。これはcolumn対応を含むdiagonal統計で、出力の交差項や意味的重要度を含まない。

| Type | R 初期8 | R 後半8 |
|---|---:|---:|
| Q | 3.168 | 1.007 |
| K | 7.821 | 1.792 |
| V | 0.215 | 0.662 |

Q/K/Vは全80stateで入力energyが同一なのに、初期Kのweight energyは入力高energy列に偏り、Vは逆に低く対応する。入力分布だけをprojection importanceへ写すとこの違いを見落とす。Wanda自体はWとAの積をすでに含むため、このRをそのまま新proxyとして足すことは正当化しない。

既存super-outlier記録も同方向：初期のchannel3848成分の出力energy比はQ43.65%、K66.07%、V0.72%（state別比の平均、交差項あり）。ただし以前、3848単独統計のdamageとの関係はlayer/type/reconstructionを調整すると消えた。この旧否定結果を撤回しない。

## Results 5 — Maskingによる内部変化はあるが、損傷への橋渡しは弱い

Clean/corruptedの平均score順序はtype内でもrho=.9967–1.0000。分布中央部の後半増加もcleanに存在する（Q median6.34×、V12.92×、attention output19.28×）。この大きなdepth patternをmaskingの有無だけに帰属できない。

一方、channel energyを総和1へ正規化したTV距離はclean/corrupted間で変化する。初期Q/K/V平均.161はblock0の.652に強く影響され、block0除外なら.091。Attention outputはquartile順.189/.097/.077/.141、FF-outは.162/.116/.123/.170。Low/high mask状態間のFF-out TVは.338–.514とさらに大きい。全体順位が似ていても内部構成が同じとは限らない。

しかしnative module-only50 signed CEとのlayer/type調整後rank相関は、clean/corrupted TVで+.084、文ごとのTV平均で+.040、state profile変動で−.053、mask主効果分散比で−.016と弱い。この結果からmask sensitivity/timestep weightingを新しい方法として進める根拠は得られていない。Signed CEは既存198/224 intervalsが0を含む不確実なtargetであり、弱い相関を機能無関係の証明とも扱わない。

## Results 6 — 既存交換結果の支持と反例を再点検

- Cross-depth/same-type4組は後半を保護する方向が全てpoint estimateで有利。最大効果のV組を除いても平均差は+0.00022567、探索的article95%CI[+0.00007128,+0.00037998]。ただしdepth以上の情報を示す比較ではない。
- Same-layer4組の逆方向の平均差は、block23 V↔attention-output組への依存が大きい。その組を除くと−0.00001540、95%CI[−0.00015737,+0.00011539]。従来の「軸で方向が異なる」は記述として残るが、type間で一般的な逆法則があるとはいえない。
- 全12組の個別article CIを同時性に配慮して99.5833%にすると、0を含まないのはpair03/08/11の3組。4/4などのpoint signsを全て確定効果として扱わない。
- 実際の境界weightをdense Aで測ったscore²は全12組でshape方向を好むが、NELBO point signが一致するのは6/12。明確な反例pair11ではscore²差+10536.29に対してNELBO差−0.0012573。これはdiagonal energyだけでは機能損傷を一意に説明できない具体例。選択済み12組なので一般的な予測精度50%とは呼ばない。

## Hypothesis A — 保護すべき対象は、すでに残る尾部から追加で失われる非tail接続へ移る

**観測からの導出：** 初期attention系は尖りが強い。一方、後半は中央部まで大きくなり、同じ50%で失うdiagonal energy割合も増える。大tailは両Uniformでも交換でも保存されており、実際の性能差は非tail部分の変更に伴って生じている。

**仮説：** このLLaDA50%近傍では、初期の大tailをさらに保護するための予算より、後半の広がった非tail接続を維持する予算の機能的価値が高い場合がある。分布全体のtail量を「追加保護の必要性」へ変換すると、既に満たされた保護に予算を配ってしまう可能性がある。

**支持の範囲：** tail countでは説明できないことは直接確認済み。非tailの分布構造が追加損傷の原因であることや、全type・全sparsityへの一般化は未確認。特にFF-outは単純な平坦化の反例。

**反証条件：** tailを同一に保った比較で、非tail分布・実際の交換境界に関する情報がdepth/typeと通常reconstructionを超えて説明しない、または独立文書・新しい同type組で後半保護が再現しない。

**Proxyへの含意：** 全体mean/varianceの置換より、「候補maskを少し変更した際に失う部分」を測定対象にする根拠。ただし境界score²だけではpair11を説明できず、完成したproxyではない。既存signed-gain allocatorやlog-varianceを再提案しない。

## Hypothesis B — 生のscoreで異なる計算経路を競わせると、出力側の非tail接続を不足させる

**観測からの導出：** 同一入力でもQ/K/Vのweight-energy対応は大きく異なる。Layer-globalでは初期FF-outが77%削られる一方でFF-proj/upは30%程度となり、出力側の非tail energy除去が大きく増え、PPLは悪化した。単一のscore単位で経路間の保護必要性を交換できる保証がない。

**仮説：** Raw scoreの小さいprojectionにも、残差へ計算結果を渡すために必要な接続が広く存在する。入力側の大score接続を増やすことで、出力側の非tail接続不足を補えるとは限らない。現状のlayer-globalの不利益の一部は、この経路間の偏りに起因する可能性がある。

**反証条件：** Layer-globalのlayer budgetを保ち、既存FF-out/attention-outputの予算偏りだけを戻しても損傷が改善しない、または改善がrow分布だけで全て説明される。既存のprojection allocation方法の成否をそのままこの仮説の検証にしない。

**Proxyへの含意：** Projectionを名指しで保護する定数ではなく、その入力分布がweightを通って何を変えるかという対応を測る必要性。まだ実際のoutput geometryに接続する統計が不足しており、Rや単一channelを新scoreとしない。

## Priority and decision

1. 主仮説はA。今回の新しい全mask・境界統計と既存介入が直接つながる。
2. Bは同時に説明すべき経路の問題。Aから「全レイヤーを同じ一変数で並べればよい」と飛躍しないための制約でもある。
3. Mask内部変化は観測事項として保持するが、損傷との結び付きが弱いため主仮説にしない。
4. FF-out末尾の再集中は反例として保持する。D31の方向×receiver機構は既存証拠であり、新発見として再包装しない。
5. 現時点の証拠は **このLLaDAでのpruning解釈** を支える。AR対照がなく、cleanでも大局パターンが残るので、DLM固有性や論文の新規性はまだ主張できない。

## Verification / limitations

- 224 features・damage source identity、24 boundary files、400 article checkpoints、original aggregate bootstrapを別scriptで確認。入力8×10とshared Q/K/V・MLP入力も確認。
- 全mask passでは6 checkpoint hashes、448 mask file/digest hashes、各count、両方exact global budget、dense score mean/RMS/tail countの過去統計再現を確認。
- 探索的再利用であり、事後に候補を比較した。CIは固定pair集合に条件づけたarticle bootstrap。新しいholdoutでもAR比較でもない。
- 分布、diagonal energy、reconstruction、最終NELBOを同一視しない。旧65% capacity関連は歴史的補助証拠に留める。
- 分析実装で8×5×2を仮定したextensionの検証assertが一度停止した。原manifestの8×10×1へ修正して実行済み。入力・モデル・既存実験設定を変更していない。
- 新規GPU実験は実行せず、仮説を設定してこの解析を完了した。

## Artifacts

- `distribution_structure.png`: six-panel distribution/channel figure.
- `projection_features.csv`, `sequence_details.json`, `results.json`: all exploratory descriptive features and associations, including weak/negative observations.
- `boundary_statistics.json`: actual changed weights, projection-energy alignment and historical read decomposition.
- `mask_audit.csv`, `mask_audit.json`: all448 existing-mask count/energy summaries and hashes.
- `pair_uncertainty.json`: per-pair uncertainty and leave-one-pair robustness.
- `verification.json`, `mask_verification.json`: independent feature/article checks and all448-mask aggregate checks.
- `analyze.py`, `extend.py`, `audit_masks.py`, `verify.py`: reproducible CPU analysis.
