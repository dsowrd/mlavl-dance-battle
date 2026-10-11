# MLAVL 재현·개선 → 댄스배틀 승패 판정 프로젝트

이 파일은 Claude Code가 이 프로젝트를 이어받기 위한 인수인계 문서다. 작업 전에 끝까지 읽고, 결정이 바뀌면 이 파일을 갱신한다.

- 작성: 2026-10-11 (정현도, 학번 2025198001)
- 과목: 심층학습 (충북대 지능로봇공학과, 2026-2학기, 화 5–7교시, 황영배 교수님)
- 최종 목표: 댄스배틀 영상(두 댄서의 라운드)을 보고 **누가 이겼는지 판정**하는 모델

---

## 1. 과목 요구사항과 일정

과제 요구사항:
- 최근(2022–2026) CVPR/ICCV/ECCV/NeurIPS/AAAI/ICLR 논문 2편 발표
- 발표자료 7쪽 내외, 7분 이내 (넘으면 감점)
- 중간·기말 발표: 발표한 논문 중 하나를 **구현하고 개선**
- 평가: 출석 20%, 논문 발표와 프로젝트 80%

주차별 일정 (1주차 = 9/1, 화요일 수업):

| 날짜 | 수업 | 상태 / 할 일 |
| --- | --- | --- |
| 10/6, 10/13 | 논문 발표 1 | MLAVL 발표 (자료 완성·제출됨) |
| 10/20 | 논문 survey | 두 번째 논문 확정 (Bailando, CVPR 2022 유력) |
| 10/26–10/31 | — | **학회 출장(삿포로). 사람은 자리를 비움 → 장시간 학습을 이때 돌린다** |
| 10/27 | 논문 발표 2 | 출장과 겹침 → 11/3 발표로 교수님께 조정 요청 예정 (미확정) |
| 11/3 | 논문 발표 2 | 두 번째 논문 발표 |
| 11/10, 11/17 | **중간발표** | 재현 결과 + 1차 개선 (발표 순서 미정) |
| 11/24 | 논문 구현 | 배틀 데이터 구현 중간 점검 |
| 12/1, 12/8 | **최종발표** | 배틀 승패 판정 결과 |

---

## 2. 기반 논문: MLAVL

- Xu et al., "Language-Guided Audio-Visual Learning for Long-Term Sports Assessment", CVPR 2025
- 공식 코드: https://github.com/XuHuangbiao/MLAVL (GDLT, CVPR 2022 기반)
- 과제: 몇 분짜리 연기 영상 + 배경음악 → 심판 점수 회귀 (Spearman ρ ↑, MSE ↓)
- 핵심: 경기 규정의 동작 용어를 텍스트 프롬프트로 넣어 audio-visual feature를 안내

구조 (frozen backbone 위에 4개 모듈만 학습, 학습 파라미터 약 3.82M):

1. **S²CE**: 모달별 token projection(1×1 Conv 768→384→256) + V·A가 **가중치를 공유**하는 Transformer encoder + residual
2. **MAG²**: 동작 텍스트 노드 M개 + 클립 노드 T개 그래프. GraphSAGE 2층(`SAGEConv` d→d/2→d). 클립↔클립, 텍스트↔텍스트 완전연결, 텍스트→클립 단방향. V용·A용 그래프 별도
3. **AVCF**: 전역 Cross-Temporal Relation Decoder(Q=영상, K·V=음악, 2층) + 클립별 concat → 1×1 Conv 블록, 둘을 더함
4. **DPG**: 등급 텍스트를 query로 쓰는 Performance Grading Transformer(3층, V/AV 공유). V는 N=6등급, AV는 2N=12등급("with/without fitting music"). 등급 확률 × 고정 등급 가중치 (n−1)/(N−1)의 합 → 학습되는 α(초기 0.5)로 두 점수 혼합

손실 (코드 `models/loss.py` 기준): `10·MSE + alpha·Σ HardTriplet(6개 임베딩) + 0.01·CE(label smoothing 0.1)`

---

## 3. 공식 코드에서 확인한 사실과 함정 (반드시 처리)

직접 읽고 확인한 내용이다.

- **`main.py`가 `FS1000Dataset`만 쓰도록 하드코딩되어 있다.** `datasets.py`에는 `FisVDataset`, `RGDataset`, `LOGODataset`도 있으니 `--dataset` 인자를 추가해 클래스를 고르게 고칠 것. README의 예시 명령도 FS1000과 RG 인자가 섞여 있어 그대로 믿으면 안 된다.
- **CLIP 의존 제거:** `models/model.py`
  - L127 `self.CLIP = returnCLIP(...)`: ViFi-CLIP 가중치(`weights/k400_clip_complete_finetuned_30_epochs.pth`)를 요구한다.
  - L149 `text_feature = self.CLIP()`
  - 저장소에 미리 계산된 텍스트 feature가 있다: `FS_text_feature.npy`(피겨 FS1000·Fis-V 공용), `RG_text_feature.npy`, `LOGO_text_feature.npy`.
  - 확장자는 .npy지만 실제로는 **`torch.save` zip 형식**이다. `torch.load(path, weights_only=True)`로 읽는다(`np.load` 불가).
  - L127을 끄고, L149를 파일 로드로 바꾸고, 텐서를 `register_buffer`로 한 번만 올린다.
- **텍스트 feature 순서와 `--n_text`:** `[동작-V M개][동작-A M개][AV 등급 2N개][V 등급 N개]`
  - FS: 42+42+12+6 = 102
  - RG: 32+32+12+6 = 82
  - **RG는 반드시 `--n_text 32`와 `--action_list ./action-label_RG.csv`.** 기본값은 42(피겨)다.
- `.cuda()` 하드코딩 (`w1`, `weight_AV/V`, `edge_index` 등) → GPU 필수
- `setup_seed`가 `torch.use_deterministic_algorithms(True)`를 켠다. PyG scatter 연산에서 에러가 나면 기록하고 끄거나 `warn_only=True`로 바꿀 것.
- MAG²가 배치 안에서 샘플마다 파이썬 루프로 GCN을 돈다(느림). 재현 단계에서는 그대로 두고, 필요하면 나중에 batch graph로 최적화한다.
- 체크포인트는 `./ckpt/{model_name}_best.pkl`(ρ 최고)과 `_best_mse.pkl`(MSE 최고)에 저장된다. 논문 수치 비교는 둘 다 기록한다.
- 의존성: `torch`, `torch_geometric`, `tensorboardX`, `timm`, `scipy`, `pandas` (+ CLIP 쓰면 `ftfy`, `regex`)

### 논문 설정값 (본문 4.1절 + 보충자료)

| 항목 | FS1000 | Fis-V | RG |
| --- | --- | --- | --- |
| clip 수 (`--clip-num`) | 95 | 124 | 68 |
| lr (논문) | 3e-4 | 3e-4 | 1e-4 |
| dropout | 0.3 | 0.7 | 0.3 |
| λ (TL / CE / MSE) | 1.0 / 0.01 / 10 | 0.5 / 0.1 / 10 | 1.0 / 0.1 / 10 |
| ξ (점수 정규화) | TES 130, PCS 60, 나머지 10 | TES 45, PCS 40 | 25 |
| epoch | TES 510, PCS 540, SS 390, TR 380, PE 400, CO 450, IN 450 | TES 420, PCS 440 | Ball 620, Clubs 340, Hoop 450, Ribbon 710 |

- 공통: d=256, N=6, margin δ=1, batch 64, Adam(wd 1e-4), cosine(decay 0.01)
- **README 예시 lr은 1e-2로 논문(3e-4/1e-4)과 다르다.** 논문 값으로 먼저 돌리고 차이를 기록한다.
- 위 λ 표의 매핑은 "λ1,λ2,λ3 = 10, 1.0/0.5/1.0, 0.01/0.1/0.1"이라는 논문 문장과 코드를 맞춰 해석한 것이다.
  - 코드는 CE 가중치가 0.01로 고정되어 있다. Fis-V·RG에서는 0.1로 바꿀 인자를 추가할 것.

### 재현 목표 수치 (논문 Table 1, 2, 3)

| 데이터 | ρ | MSE |
| --- | --- | --- |
| RG | Ball 0.826 / Clubs 0.829 / Hoop 0.871 / Ribbon 0.866 / **Avg 0.849** | 5.57 / 4.20 / 4.11 / 3.99 / **Avg 4.47** |
| FS1000 | TES 0.92 / PCS 0.89 / SS 0.90 / TR 0.90 / PE 0.88 / CO 0.89 / IN 0.88 / **Avg 0.90** | 64.89 / 6.39 / 0.23 / 0.24 / 0.50 / 0.25 / 0.26 / **Avg 10.39** |
| Fis-V | TES 0.766 / PCS 0.863 / Avg 0.823 | 19.44 / 7.17 / 13.31 |

- 평균 ρ는 AQA 관례상 Fisher z 변환 평균일 가능성이 높다. 산술 평균과 둘 다 계산해 기록한다.

### feature 다운로드 위치 (README 기준)

- RG, Fis-V (RGB VST, Audio AST, Flow + 라벨): PAMFN 저장소 https://github.com/qinghuannn/PAMFN
- FS1000 (RGB, Audio + 라벨): Skating-Mixer 저장소 https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating
- LOGO: https://github.com/dx199771/Interpretability-AQA
- `RGDataset`은 `{video_path}/{Ball|Clubs|Hoop|Ribbon}_rgb_VST.npy` 형식을 기대한다. 받은 파일 구조와 맞는지 먼저 확인할 것.

---

## 4. 실행 환경: Server 1만 사용

- 사양 (2026-10-11 직접 확인): RTX A6000 48GB × 4 · RAM 251GB · 루트 SSD 409GB(`/`, 여유 약 25GB) + HDD 3.7TB(`/dev/sdb1`, ext4, **미마운트**) · Ubuntu 20.04.6 · 드라이버 535.230.02(CUDA 12.2) · 기본 nvcc 11.8, `/usr/local`에 11.3/11.6/11.7/11.8/12.1/12.6/12.8 toolkit (위치: E8-10 B03)
- **GPU 2는 하드웨어 에러로 사용 금지. GPU 0, 1, 3만 사용한다.** (nvidia-smi 상으로는 정상처럼 보이지만 부팅 로그에 GPU 2(PCI 99:00) Xid 62 기록이 있다. 규칙 유지)
- 모든 실행에 아래 두 줄을 붙인다:

  ```bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID      # nvidia-smi 번호 = PyTorch 번호
  CUDA_VISIBLE_DEVICES=0 python main.py ... # 0, 1, 3 중 하나. 절대 2 포함 금지
  ```

- GPU 배분: 0 = 재현, 1 = 개선·ablation, 3 = 배틀 feature 추출(비면 재현 추가)
- 연구실 규칙:
  - 장시간 작업 전 Teams › IRL CBNU › General › **Schedule Server Usage** 탭에 예약한다. 예약은 사람이 하므로 필요하면 요청할 것.
  - 환경은 **Conda 또는 Docker만** 쓴다(venv 금지).
  - CUDA를 추가 설치할 때는 드라이버를 체크 해제하고 toolkit만 설치한다.
  - 결과와 데이터는 서버에만 두지 말고 **NAS에도 백업**한다.
- 저장공간:
  - 루트 SSD(409GB, 여유 약 25GB): conda 환경(`mlavl` 5.5GB), 코드, 체크포인트. **여유가 거의 없다.** 공개 feature도 HDD에 둔다
  - HDD 3.7TB(`/dev/sdb1`) → **`/data`에 마운트됨 (2026-10-11, 정현도).** 단 여유는 111GB뿐이다(다른 사람 자료 `/data/irl`, `/data/docker-data`가 3.3TB 차지. 건드리지 않는다). **fstab에 없어서 재부팅하면 풀린다.** 등록하려면 `UUID=2f54f02a-274b-43d3-9ea3-56045d5063da /data ext4 defaults,nofail 0 2` 한 줄을 sudo로 추가
  - 우리 데이터 루트 `MLAVL_DATA=/data/mlavl`: `features/{RG,FISV,FS1000}`, `videos/`, `frames/`, `datasets/{brace,olympics2024}`, `ckpt_backup/`
  - NAS: 서버에 마운트되어 있지 않다(fstab·mount에 NFS/CIFS 없음). 주소를 받으면 확인
- 외부 인터넷: github, pypi, PyTorch/PyG wheel 인덱스, YouTube, Google Drive 모두 접근 가능 (프록시 없음)
- 공유 서버이므로 장시간 학습은 `tmux` 또는 `nohup`으로 띄우고, Claude Code 세션이 끝나도 계속 돌게 한다.

실제 환경 (2026-10-11 구성 완료, 스모크 테스트 통과):

```bash
# conda env: /home/irl/anaconda3/envs/mlavl  (Python 3.10.22)
# torch 2.4.1+cu121, torchvision 0.19.1, torchaudio 2.4.1   ← 드라이버가 CUDA 12.2라 cu124 대신 cu121
# torch_geometric 2.8.0 (+ pyg_lib, torch_scatter, torch_sparse), tensorboardX, timm, scipy, pandas
source scripts/env.sh                      # CUDA_DEVICE_ORDER, MLAVL_PY, MLAVL_DATA, GPU 2 차단
CUDA_VISIBLE_DEVICES=0 $MLAVL_PY main.py ...
```

- **주의: 비대화형 셸(Claude Code 백그라운드 포함)에서 `conda activate` 뒤의 `pip`는 `~/.local/bin/pip`(시스템 Python 3.8)로 풀린다.** 반드시 `$MLAVL_PY -m pip ...` 또는 `/home/irl/anaconda3/envs/mlavl/bin/python`을 명시한다
- `torch.use_deterministic_algorithms(True)` 상태에서 PyG SAGEConv forward/backward가 에러 없이 동작함을 확인했다(스모크 테스트)
- 스모크 테스트의 학습 파라미터 수는 6.72M으로 논문의 3.82M과 다르다. 집계 기준 차이로 추정. Phase 1에서 모듈별로 세어 기록할 것

---

## 5. 작업 계획

### Phase 0 · 세팅 (10/12–10/18)

- [x] 서버에 프로젝트 폴더 생성. 7절 방식대로 MLAVL을 clone하고 **정현도 개인 GitHub의 private 저장소**에 연결 (`/home/irl/Workspace_Hyundo/mlavl-dance-battle`, origin = `dsowrd/mlavl-dance-battle`, upstream = XuHuangbiao/MLAVL, 작업 브랜치 `dev`)
- [x] README의 출처·크레딧 섹션(7절)을 첫 커밋부터 넣는다 (2026-10-11)
- [x] conda 환경 구성, `torch.cuda.is_available()`로 GPU 0/1/3 확인 (2026-10-11, 4절 참조)
- [x] 3절의 함정 처리: `--dataset` 인자, CLIP 제거 + 텍스트 feature 로드(`--text_feature`, `--use_clip`), CE 가중치 인자(`--alpha_ce`). `scripts/smoke_test.py`로 RG/FS1000 설정 forward+backward 확인
- [x] `.gitignore`, `experiments/EXPERIMENTS.md`, `scripts/env.sh` 추가. 커밋돼 있던 `__pycache__` 추적 해제
- [x] 브레이킹 무브 어휘 초안 `ours/vocab/action-label_BREAKING.csv` (36 moves, `--n_text 36`). 정현도 검토 전
- [x] HDD 마운트 → `MLAVL_DATA=/data/mlavl` (2026-10-11)
- [x] BRACE release(keypoint, audio feature) + annotations, 올림픽 2024 결과집 PDF를 `/data/mlavl/datasets/`에 다운로드 (Phase 4 선행)
- [x] RG·Fis-V feature(PAMFN) 다운로드 완료 (정현도가 브라우저로 받음, 2026-10-11). `/data/mlavl/features/pamfn_release.zip`(4.0GB 원본 보관) → `pamfn_data/`에 압축 해제, `features/RG/`, `features/FISV/`에 MLAVL이 기대하는 이름으로 symlink
  - **확인된 구조**: `{App}_rgb_VST.npy` = `{video_id: (T, 1024)}`, `{App}_audio_AST.npy` = `(T, 768)`, T는 RG 68–77, Fis-V 124–302. RG 기구당 train 200 / test 50, Fis-V train 400 / test 100. flow(I3D 1024), UMT, MAST feature도 들어 있으나 MLAVL은 쓰지 않음
  - **`--in_dim 1024` 필수** (README의 768은 이 파일과 맞지 않음). `--clip-num` 68 / 124는 최단 영상 길이와 일치
  - 추가로 고친 upstream 버그 2개(`datasets.py`): RGDataset이 `args.score_range`를 안 넣어 test에서 크래시, FisVDataset이 헤더 없는 PAMFN txt의 첫 줄(영상 1, 2)을 버림

### Phase 1 · RG 재현 (10/14–10/25)

- [ ] RG 4개 기구 학습(논문 설정). 기구마다 학습 시간과 최대 VRAM을 기록
  - 2026-10-11 15:48 시작: `scripts/run_rg.sh {Ball,Clubs,Hoop,Ribbon} 0`을 tmux 세션 `mlavl_rg`(창 이름 = 기구)에서 GPU 0에 4개 동시 실행. 로그 `logs/stdout/rg_*_1011_1548.log`. 단독 실행 시 epoch당 약 5초, 4개 동시 실행 시 약 15–20초, 프로세스당 VRAM 약 1.8GB. 결과 집계는 `scripts/summarize_runs.py`
- [ ] 목표 수치와 비교표 작성 (`experiments/EXPERIMENTS.md`)
- [ ] README lr(1e-2)과 논문 lr 차이 실험 1회

### Phase 2 · Fis-V 재현 (+ FS1000은 구하면) (출장 기간 10/26–10/31에 무인 실행)

**변경(2026-10-11): FS1000 feature는 Baidu Netdisk 전용이고 정현도는 Baidu 계정이 없다. 기본은 Fis-V(TES, PCS. PAMFN 배포에 포함)로 바꾼다.** FS1000은 저자(jingfeixia708@gmail.com)에게 메일로 요청해 두고, 출장 전에 받으면 추가한다.

- [ ] Fis-V TES/PCS 학습 스크립트(`scripts/run_fisv.sh`, `--dataset fisv --clip-num 124 --dropout 0.7 --alpha 0.5 --alpha_ce 0.1`) 완성. 출장 중에는 RG·Fis-V 반복 실험(seed, lr)을 GPU 0/1/3에 큐 방식으로 배분
- [ ] (선택) FS1000 저자 메일 요청. 받으면 7개 점수 × `scripts/run_fs1000.sh`
- [ ] 돌아와서 결과 표 작성. RG 원본 영상(ACTION-NET OneDrive)을 받을 수 있는지도 이 단계에서 확인

### Phase 3 · 중간발표용 1차 개선 (11/1–11/9, 공개 feature만 사용)

원본 영상 없이 가능한 개선을 우선한다. 논문 한계 ③④에 대응한다.

- [ ] **Multi-task head**: TES·PCS(+5개 세부점수)를 한 모델에서 같이 예측해 점수 간 상관을 활용
- [ ] **고득점 과소추정 완화**: 고득점 샘플 가중 MSE, 또는 label distribution smoothing / balanced MSE
- [ ] 각 개선의 ablation 표(ρ, MSE, 고득점 구간 MSE 별도)
- [ ] 원본 영상을 구할 수 있으면 pose branch를 여기서 먼저 시도해도 됨 (Phase 5 내용)

### Phase 4 · 배틀 데이터 구축 (11/10–11/24)

승패 라벨이 붙은 공개 데이터셋은 없다. 공개 자료를 조합한다.

- **BRACE (ECCV 2022)**: https://github.com/dmoltisanti/brace
  - Red Bull BC One 1:1 브레이킹 배틀 81개 영상 (2011, 2013, 2014, 2017, 2018, 2020), 465 sequence, 1,352 segment, 댄서 64명, 3h 32m
  - 제공: 댄서별 2D keypoint(COCO-17, bbox), 구간 라벨(toprock / powermove / footwork, 댄서 ID, 연도), 음악 feature와 beat(Essentia), 샷 경계, train/test split
  - 영상은 유튜브 링크 + 다운로드 스크립트로 제공(`yt-dlp` 사용 권장). 라이선스 S-Lab 1.0(비상업). **영상은 연구용으로만 쓰고 재배포 금지**
  - 승패 라벨 없음 → 연도별 Red Bull BC One 공식 대진표 결과로 **승자를 수작업으로 붙인다**. 사람(정현도)의 확인을 거칠 것
- **2024 파리 올림픽 브레이킹 공식 결과집** (Olympics library, Results Book)
  - 배틀 64개, 심판 9명의 라운드별 투표 + 5개 항목 점수(Technique, Vocabulary, Originality, Execution, **Musicality**)
  - 라벨은 풍부하지만 전체 배틀 영상 확보 여부가 불확실(IOC 저작권). 영상을 구하면 항목별 점수 실험에 사용
- 참고: ImperialDance(AAAI 2024, 숙련도 3단계 mocap, 영상 아님), Ranked TikTok Dance(IJAIN, pairwise, 공개 여부 불명)

할 일:

- [ ] BRACE 메타데이터 + 영상 다운로드 (HDD)
- [ ] 배틀 단위로 묶기: (연도, 대진, 댄서 A, 댄서 B, 라운드별 sequence) → `battles.csv`
- [ ] 승자 라벨 `winners.csv` 작성 (출처 URL 함께 기록)
- [ ] feature 추출: MLAVL 저장소에는 추출 코드가 없으니 직접 작성한다
  - 영상: 32프레임 비중첩 클립 단위 Video Swin. **feature 차원이 `--in_dim`(768)과 맞아야 한다.** Swin-B는 1024차원이니 맞는 모델을 고르거나 `in_dim`을 바꾼다
  - 음악: AST (AudioSet). 같은 클립 경계로 정렬
  - 함성·MC 소리 때문에 비트 검출 품질을 확인한다. BRACE 제공 beat와 librosa/madmom 결과를 비교
- [ ] 데이터 분할: 같은 댄서가 여러 해에 나오므로 **연도 단위 leave-one-year-out 교차검증**으로 누수를 막는다

### Phase 5 · 배틀용 모델 (11/17–11/30)

- [ ] **MAG² 어휘 교체**: 피겨 동작 42개 → 브레이킹 무브 용어
  - 초안 예: toprock, footwork(6-step, 3-step), freeze(baby freeze, air chair), power move(windmill, flare, headspin, airflare), go-down, transition
  - **최종 목록은 정현도가 직접 확정한다** (본인이 스트릿 댄서)
  - 등급 프롬프트도 "a video of [grade] breaking performance with/without fitting music" 형태로 수정. CLIP 텍스트 feature를 다시 계산해야 하므로 이때는 CLIP text encoder(원본 CLIP 또는 ViFi-CLIP)를 다시 켜서 한 번만 추출·저장
- [ ] **Pose branch**: BRACE keypoint → bbox 정규화 + 관절 속도 → 클립 단위 작은 temporal encoder → 세 번째 modality로 S²CE에 넣거나 AVCF clip-wise 경로에 concat
- [ ] **Kick-beat feature**: 관절 평균 속도의 지역 최소점 = kinematic beat. 클립마다 음악 beat ±δ 안에 kinematic beat가 있는 비율(Bailando의 Beat Align Score 방식)
- [ ] **승패 판정 (siamese)**: 같은 모델로 ŝ_A, ŝ_B → P(A 승) = σ((ŝ_A − ŝ_B)/τ), BCE 또는 MarginRankingLoss. 표 비율(3:0, 2:1)을 알면 soft label로 사용
- [ ] 데이터가 적으므로 FS1000으로 pretrain한 가중치에서 fine-tune (점수 head는 재초기화)

### Phase 6 · 평가와 최종발표 (12/1, 12/8)

- 지표: pairwise accuracy, AUC. 표본이 작으니 bootstrap 신뢰구간 포함
- baseline:
  1. 랜덤(50%)
  2. "동작 에너지(관절 속도 합)가 큰 쪽이 이긴다" 규칙
  3. 영상만
  4. 영상 + 음악 (텍스트 없음)
  5. MLAVL 전체
  6. \+ pose / + kick-beat
- 실패 사례 분석(어떤 배틀에서 틀렸는지 영상 구간과 함께)

---

## 6. 작업 규칙

- **GPU 2 사용 금지.** 모든 실행에 `CUDA_DEVICE_ORDER=PCI_BUS_ID`와 명시적 `CUDA_VISIBLE_DEVICES`를 쓴다
- 실험마다 `experiments/EXPERIMENTS.md`에 한 줄씩 기록한다: 날짜 · git commit · 데이터/점수 · 주요 인자 · GPU · 시간 · 최대 VRAM · best ρ · best MSE · 비고
- 설정은 스크립트(`scripts/*.sh`)로 남긴다. 셸에서 즉석으로만 실행하지 않는다
- 원본 코드 수정은 커밋 단위로 하고, 수정 이유를 커밋 메시지에 남긴다
- 데이터 삭제, 대용량 다운로드(수십 GB 이상), 장시간 GPU 점유 전에는 사람에게 먼저 확인한다
- 유튜브 영상은 연구 목적으로만 사용하고 외부에 올리지 않는다
- 논문 수치와 다르게 나오면 숨기지 말고 차이와 추정 원인을 기록한다. 발표 자료의 재료가 된다

---

## 7. 저장소 관리와 출처 표기

이 프로젝트는 독창적 연구라기보다 **MLAVL을 재현하고 개선해 새 도메인(댄스배틀)에 적용하는 작업**이다. 원 저자들의 기여가 분명히 드러나도록 저장소를 관리한다.

### 어디에 두나

- **정현도 개인 GitHub 계정**에 둔다. 연구실 계정(`irlab6342`)이나 조직(`irl-cbnu`)에는 올리지 않는다
- 저장소 이름 예시: `mlavl-dance-battle` (최종 이름은 정현도가 정함)
- **처음에는 private으로 둔다.** 확인 결과 공식 MLAVL 저장소와 그 기반인 GDLT 저장소에는 LICENSE 파일이 없다. 라이선스가 없는 코드는 기본적으로 저작권이 원 저자에게 있다
- 나중에 public으로 바꾸려면 둘 중 하나를 택한다:
  1. MLAVL 저자에게 연락해 공개 허락이나 라이선스 추가를 요청한다
  2. 원본 코드는 넣지 않고, 우리 코드와 원본 대비 patch만 공개한다. 이 경우 README에 "원본을 clone한 뒤 patch를 적용"하는 방법을 적는다

### 원본 이력을 그대로 보존

```bash
git clone https://github.com/XuHuangbiao/MLAVL.git mlavl-dance-battle
cd mlavl-dance-battle
git remote rename origin upstream                  # 원 저자 저장소
git remote add origin git@github.com:<개인계정>/mlavl-dance-battle.git
git push -u origin main                            # 원 저자 커밋 이력이 그대로 올라감
git checkout -b dev                                # 우리 작업은 dev 브랜치에서
```

- 원 저자의 커밋 이력은 지우거나 squash하지 않는다. 누가 무엇을 썼는지가 이력에 그대로 남아야 한다
- `main`은 원본 상태를 유지하고, 우리 변경은 `dev`와 기능 브랜치에서 한다

### 우리 것과 원본을 구분

- 커밋 메시지 접두어:
  - `fix(upstream):` 원본 코드를 실행하기 위한 최소 수정 (예: CLIP 제거, `--dataset` 인자)
  - `feat:` 우리 개선 (multi-task, pose branch, pairwise 등)
  - `data:` 데이터 처리 스크립트, `exp:` 실험 스크립트·기록, `docs:` 문서
- 원본 파일을 고칠 때 파일 상단에 주석을 남긴다:

  ```python
  # Modified from MLAVL (Xu et al., CVPR 2025), https://github.com/XuHuangbiao/MLAVL
  # Changes: <무엇을 왜 바꿨는지 한두 줄>
  ```

- 새로 만든 모듈(pose branch, 배틀 데이터셋, pairwise 학습 등)은 별도 파일·폴더로 둔다 (예: `ours/`, `tools/battle/`)

### README 구성

맨 위 첫 문단에서 기반 연구를 먼저 밝힌다. 예:

> This repository builds on **MLAVL** (Xu et al., "Language-Guided Audio-Visual Learning for Long-Term Sports Assessment", CVPR 2025, [code](https://github.com/XuHuangbiao/MLAVL)). Most of the model code is theirs. This project reproduces their results and extends the method to 1-vs-1 dance battle judging as a course project (Deep Learning, CBNU, Fall 2026).

이어서:

1. **What is ours vs. upstream**: 우리가 추가하거나 바꾼 것 목록
2. **Reproduction results**: 논문 수치와 우리 재현 수치를 나란히 표로
3. **Acknowledgements**: 아래 크레딧 표
4. **Citation**: 원 논문 BibTeX를 먼저, 사용한 데이터셋과 방법의 BibTeX를 그다음에

### 크레딧 대상 (사용하는 것만 남긴다)

| 대상 | 쓰는 곳 | 비고 |
| --- | --- | --- |
| MLAVL (Xu et al., CVPR 2025) | 모델 코드 전체, 텍스트 feature, 동작 라벨 CSV | 기반 연구 |
| GDLT (Xu et al., CVPR 2022) | MLAVL의 기반 코드 | MLAVL README가 명시 |
| PAMFN (Zeng & Zheng, TIP 2024) | RG·Fis-V feature 배포 | |
| Skating-Mixer / FS1000 (Xia et al., AAAI 2023) | FS1000 데이터·feature | |
| Rhythmic Gymnastics (ACTION-NET, Zeng et al., ACM MM 2020) | RG 데이터셋 | |
| Fis-V (Xu et al., TCSVT 2019) | Fis-V 데이터셋 | 쓸 경우 |
| BRACE (Moltisanti et al., ECCV 2022) | 배틀 영상 목록, keypoint, beat | **S-Lab License 1.0, 비상업.** 라이선스 문구를 README에 명시 |
| Bailando (Siyao et al., CVPR 2022) | Beat Align 방식 | 쓸 경우 |
| CLIP / ViFi-CLIP, Video Swin, AST | 텍스트·영상·음악 feature | |
| 2024 Olympic Breaking Results Book | 승패·항목 점수 라벨 | 쓸 경우, 출처 URL |
| Red Bull BC One 공식 결과 | 승자 라벨 | 연도별 출처 URL을 `winners.csv`에 기록 |

MLAVL BibTeX (공식 README에서 그대로 가져옴):

```bibtex
@InProceedings{Xu_2025_CVPR,
    author    = {Xu, Huangbiao and Ke, Xiao and Wu, Huanqi and Xu, Rui and Li, Yuezhou and Guo, Wenzhong},
    title     = {Language-Guided Audio-Visual Learning for Long-Term Sports Assessment},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {23967-23977}
}
```

- 나머지 BibTeX는 **각 논문의 공식 페이지나 저장소에서 복사한다.** 기억으로 쓰지 않는다

### 저장소에 올리지 않는 것 (`.gitignore`)

- feature 파일(`*.npy` 중 대용량), 원본 영상, 프레임, 체크포인트(`ckpt/`), 로그(`logs/`), 다운로드한 데이터셋
- 예외: 원 저장소에 원래 들어 있던 텍스트 feature(`*_text_feature.npy`)와 라벨 CSV는 원본 그대로 유지
- 우리가 직접 만든 라벨(`winners.csv`, `battles.csv`)과 무브 어휘 목록은 올린다. 각 행에 출처를 남긴다

### 발표 자료에서도

- 중간·최종 발표 슬라이드에서 "MLAVL 기반 재현 + 개선"임을 첫 장에 밝히고, 원 논문의 그림과 수식을 쓸 때 출처를 표기한다

---

## 8. 아직 정해지지 않은 것

- [ ] 두 번째 논문 발표일 11/3 조정: 정현도 의견은 "가능할 것 같다"(2026-10-11). 교수님 확정 대기
- [ ] 중간·기말 발표 순서(11/10 vs 11/17, 12/1 vs 12/8): 미정. **계획은 빠른 날짜(11/10, 12/1)를 마감으로 가정**한다
- [ ] RG·FS1000 원본 영상 확보 가능 여부 → pose branch를 공개 벤치마크에서도 할지 결정 (조사 중, 결과는 아래 "조사 결과"에 기록)
- [x] 외부 인터넷: 가능 (4절). NAS: 미마운트, 주소 필요. HDD: `/dev/sdb1` 미마운트, 사람이 sudo로 마운트해야 함 (4절)
- [ ] 올림픽 배틀 전체 영상 확보 가능 여부 (조사 중)
- [ ] 브레이킹 무브 어휘 최종 목록: 초안 `ours/vocab/action-label_BREAKING.csv` 작성됨, 정현도 검토 필요
- [x] GitHub: 계정 `dsowrd`, 저장소 `mlavl-dance-battle`(private), 인증은 SSH deploy key(`~/.ssh/mlavl_deploy`, alias `github-mlavl`). 로컬 git identity는 `dsowrd <dsowrd123@gmail.com>`로 설정(저장소 로컬 config). GitHub 계정 이메일과 다르면 커밋이 계정에 연결되지 않으니 확인
- [x] 저장소 공개 여부: **당분간 private 유지** (정현도, 2026-10-11). public 전환과 저자 연락은 나중에 다시 결정

### 조사 결과 (2026-10-11, 웹 조사. 다운로드는 아직 하지 않음)

**RG·Fis-V feature (PAMFN)**
- 유일한 배포처는 OneDrive 1개 링크: https://1drv.ms/u/s!ApyE_Lf3PFl2itlpIvyjNzcQ0TlsPA?e=biQ4bw (압축 해제 후 `data/`, `pretrained_models/`). 크기 미확인(수 GB 추정)
- **서버에서 curl/API로는 받을 수 없다(1drv.ms → onedrive.live.com 403, shares API 401).** 브라우저가 필요하다: 사람이 노트북에서 받아 `scp`로 HDD에 올리거나, Claude in Chrome 확장으로 받는다
- 파일 구조(PAMFN 코드 기준): `{Ball,Clubs,Hoop,Ribbon}_rgb_VST.npy`, `_flow_I3D.npy`, `_audio_AST.npy`, `FISV_rgb_VST.npy` 등. 각 .npy는 `{video_id: array}` dict(`allow_pickle=True`). 라벨 `data/RG/{train,test}.txt`, `data/FISV/{train,test}.txt` (헤더 1줄, 공백 구분). MLAVL `RGDataset`과 호환
- **주의: PAMFN 설정은 VST(RGB) 1024차원, AST 768차원이다.** MLAVL README는 RG에 `--in_dim 768`을 쓴다. 받은 뒤 실제 shape를 확인하고 `--in_dim`을 맞춘다(`proj_audio`는 768 고정)

**FS1000 feature (Skating-Mixer)**
- Baidu Netdisk만 제공: https://pan.baidu.com/s/1SGbvK6vDGR7ZP0PxakUO7g?pwd=9tma (Baidu 계정 필요, Google Drive 미러 없음). 크기 미확인
- 라벨 txt: `id TES PCS SS TR PE CO IN factor` 9열, 영상별 `{id}.npy`. MLAVL `FS1000Dataset`과 호환
- **Baidu 계정이 없으면 Phase 2(FS1000)가 막힌다.** 대안: 저자(jingfeixia708@gmail.com)에게 메일로 요청, 또는 Fis-V(PAMFN에 포함)로 Phase 2를 대체

**원본 영상**
- RG: ACTION-NET README의 OneDrive 링크 https://1drv.ms/u/s!ApyE_Lf3PFl2issDbaK99shfZRKchg?e=fdd2eO 가 영상으로 추정(미검증, 역시 브라우저 필요). 유튜브 ID 목록은 없다
- FS1000: 유튜브 ID·아카이브 없음. 저자 메일 요청만 가능
- 올림픽 2024 브레이킹: olympics.com에 전체 리플레이(B-Boys Final 2:38, Round Robin, B-Girls 동일)가 있지만 **스트리밍 전용·로그인·지역 차단(한국은 중계권자 있어 차단 가능성 높음)·다운로드 불가**. 공식 유튜브 전체 배틀 업로드 없음. → 영상 확보는 사실상 어렵다. 항목별 점수 실험은 보류

**올림픽 2024 결과집**: WDSF에서 공개 PDF 확보 가능 https://cdnb.worlddancesport.org/legacy-results/58801/OG2024_BKG_B99_BKG.pdf (0.9MB, 89쪽, HTTP 200 확인). 배틀·라운드별 심판 9명 투표와 5개 항목(Technique/Vocabulary/Originality/Execution/Musicality) 포함. Olympedia에도 배틀별 투표 페이지 있음

**BRACE**: GitHub release `af_v1.0`에 `dataset.zip`(keypoint 132MiB), `manual_keypoints.zip`(17MiB), `audio_features.zip`(1.17GiB). 저장소 `annotations/`에 `segments.csv`, `sequences.csv`, `sequences_{train,test}.csv`, `audio_beats.json`, `shot_boundaries.json`, `videos_info.csv`(81개 유튜브 URL, 1080p 29.97fps; 연도별 2011:8, 2013:15, 2014:14, 2017:15, 2018:15, 2020:14). 다운로드 스크립트는 없고 `yt-dlp`로 직접 받는다(약 10–20GB 추정, HDD)

**Red Bull BC One 승자 라벨**: 위키백과 "Red Bull BC One" 한 페이지에 6개 연도 대진표 전부 있음 (https://en.wikipedia.org/wiki/Red_Bull_BC_One, wikitext는 `action=raw`로 받아 `Round16`/`bracket`/`Round8` 템플릿을 파싱). 챔피언: 2011 Roxrite, 2013 Hong 10, 2014 Menno, 2017 Menno, 2018 Lil Zoo / B-girl Ami, 2020 Shigekix / B-girl Kastet. BRACE 유튜브 제목에 두 댄서와 라운드가 들어 있어 영상 단위로 교차 확인 가능. redbull.com 리캡으로 2차 검증
