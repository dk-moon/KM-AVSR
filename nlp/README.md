# 🇰🇷 한국어 형태소 기반 분할 단어(morpheme-based subword) 토크나이저 구축

- **Okt 형태소 분석기**를 이용해 `input.txt`에 들어있는 문장을 **형태소 단위로 분리**한 뒤, corpus를 기반으로 **SentencePiece**를 학습하여 **분할 단어(subword) 단위 토크나이저**를 생성하는 과정

## ✨ 왜 형태소 기반으로 분리할까?

- 한국어는 교착어(agglutinative language)로, 단어가 접사나 어미가 붙어 다양한 형태로 변화합니다.
- 이를 그대로 subword 학습에 넣으면, 의미 단위가 아닌 작은 음절 수준에서 잘려 불필요한 subword가 많이 생성됩니다.
- 형태소 분석(Okt 등) 을 통해 먼저 의미 단위로 분리하면, 더 직관적이고 일관된 subword 단위를 구성할 수 있습니다.

## ✨ SentencePiece에서 BPE를 사용하는 이유

- BPE(Byte Pair Encoding)는 자주 등장하는 문자 혹은 subword 쌍을 반복적으로 병합하여 서브워드를 구성합니다.
- BPE만 사용하면 형태소 분석 없이도 통계적으로 자주 등장하는 단위들을 subword로 자동 생성할 수 있습니다.
- 하지만 의미 단위 고려는 하지 않기 때문에, 한국어 같이 복잡한 어형 변화를 가진 언어에서는 의미 기반 subword가 덜 정확할 수 있습니다.

## ✨ 왜 Okt와 BPE를 같이 사용하나?

- 형태소 분석(Okt)을 통해 먼저 의미 단위로 분리 ⇒ 의미를 유지한 corpus 생성
- 그 후 BPE를 적용 ⇒ 형태소 단위에서 추가로 자주 등장하는 결합 패턴을 subword로 재구성
- 이 과정을 거치면 의미 단위 기반 + 통계 기반 subword 학습을 동시에 활용할 수 있습니다.

|구분|내용|
|:---|:---|
|**형태소 기반 분리**|의미 단위 분리, 한국어 특성 고려|
|**BPE 사용 이유**|자주 등장하는 패턴 기반 subword 학습|
|**둘을 결합하는 이유**|의미 기반 + 통계 기반 subword 결합 최적화|


---

## ✅ 과정 요약

1️⃣ `input.txt`에 한 줄씩 저장된 문장을 **Okt 형태소 분석**  
2️⃣ `morpheme_input.txt`에 **형태소 단위로 띄어쓰기된 문장** 저장  
3️⃣ `morpheme_input.txt`를 기반으로 **SentencePiece 인코더 학습**

---

## 📜 라이브러리 설치

### 📦 형태소 분석 (Okt)

```bash
uv add konlpy
```

> ⚠️ macOS의 경우 java가 필요하므로, 설치되지 않은 경우
```bash
brew install openjdk
sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
```

### 📦 서브워드 토크나이저 (SentencePiece)

```bash
uv add konlpy
```

## 💻 실행

### 형태소 분석기

```bash
python morpheme_process.py
```

- 결과
    > `input.txt`
    ```Planetext
    오늘 날씨가 정말 좋네요.
    내일은 등산을 갈 예정입니다.
    같이 가실래요?
    ```

    > `morpheme_input.txt`
    ```Planetext
    오늘 날씨 가 정말 좋네요 .
    내일 은 등산 을 갈 예정 입니다 .
    같이 가실래요 ?
    ```

### 서브워드 토크나이저 (SentencePiece)

```bash
python train_spm.py
```

## 데이터 활용

- 한국어 문장 데이터셋
    - AI Hub 말뭉치 데이터 활용
    - 립 리딩(입 모양) 발화 예측 데이터 활용
