# 한국어 형태소 기반 분할 단어(morpheme-based subword) 토크나이저 구축

- **Okt 형태소 분석기**를 이용해 `input.txt`에 들어있는 문장을 **형태소 단위로 분리**한 뒤, corpus를 기반으로 **SentencePiece**를 학습하여 **분할 단어(subword) 단위 토크나이저**를 생성하는 과정

---

## ✅ 과정 요약

1️⃣ `input.txt`에 한 줄씩 저장된 문장을 **Okt 형태소 분석**  
2️⃣ `morpheme_input.txt`에 **형태소 단위로 띄어쓰기된 문장** 저장  
3️⃣ `morpheme_input.txt`를 기반으로 **SentencePiece 인코더 학습**

---

## ✅ 라이브러리 설치

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

## ✅ 실행

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
