# Okt 형태소 분석

- `input.txt`에 들어있는 문장을 Okt 형태소 분석기로 분석하여, 형태소 단위로 분리된 문장을 `morpheme_input.txt`에 저장하는 과정

## 라이브러리 설치

```bash
uv add konlpy
```

> ⚠️ macOS의 경우 java가 필요하므로, 설치되지 않은 경우
```bash
brew install openjdk
sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
```

## 실행
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
