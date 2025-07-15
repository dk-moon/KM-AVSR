from konlpy.tag import Okt
from tqdm.auto import tqdm

# 형태소 분석기 초기화
okt = Okt()

input_file = "input.txt"
output_file = "morpheme_input.txt"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    
    for line in tqdm(fin):
        line = line.strip()
        if not line:
            continue  # 빈 줄은 건너뜀

        # 형태소 단위 분리
        morphs = okt.morphs(line)

        # 형태소를 공백으로 연결
        fout.write(" ".join(morphs) + "\n")

print(f"✅ 형태소 분석 완료! -> {output_file}")