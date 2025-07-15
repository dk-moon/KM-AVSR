import sentencepiece as spm
import os

# -------------------------
# ✅ 설정값
# -------------------------
n_units = 1207               # vocab_size(small vocab(sentence) : 200~500, middle vocab(subword) : 1,000~8,000, large vocab(word) : 16,000~32,000)
model_type = "bpe"      # model_type(unigram, bpe, word, char)

# 디렉토리 및 파일명 설정
output_dir = model_type
os.makedirs(output_dir, exist_ok=True)

dict_path = f"{output_dir}/{model_type}{n_units}_units.txt"
model_prefix = f"{output_dir}/{model_type}{n_units}"

input_file = "morpheme_input.txt"   # 형태소 기반 corpus

# -------------------------
# ✅ SentencePiece 학습
# -------------------------
spm.SentencePieceTrainer.Train(
    input=input_file,
    vocab_size=n_units,
    model_type=model_type,
    model_prefix=model_prefix,
    input_sentence_size=100000000,
    character_coverage=1.0
    # ✅ 삭제: unk_id=1
)

print("✅ SentencePiece 모델 학습 완료")
print(f"모델 파일: {model_prefix}.model")
print(f"Vocab 파일: {model_prefix}.vocab")

# -------------------------
# ✅ Dictionary 파일 생성
# -------------------------
# <unk> 항목 초기화
with open(dict_path, "w", encoding="utf-8") as f:
    f.write("<unk> 1\n")

# 모델 로드
sp = spm.SentencePieceProcessor()
sp.Load(f"{model_prefix}.model")

# corpus를 토큰화 → 단어별 개별 줄 → sort & uniq
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

token_set = set()

for line in lines:
    line = line.strip()
    if not line:
        continue
    pieces = sp.EncodeAsPieces(line)
    token_set.update(pieces)

# 정렬
token_list = sorted(list(token_set))

# 사전 파일에 기록
with open(dict_path, "a", encoding="utf-8") as f:
    for idx, token in enumerate(token_list, start=2):  # 1번은 <unk>, blank는 0
        f.write(f"{token} {idx}\n")

print(f"✅ Dictionary 생성 완료 -> {dict_path}")