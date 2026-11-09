# How Mesh-TensorFlow Works

**데이터 병렬은 batch 차원을 나누는 레이아웃 하나일 뿐이고, 어떤 차원이든 이름을 붙여 메시 축에 나누면 같은 SPMD 프로그램이 모델 병렬이 된다 — 통신은 나눈 차원을 더해 없애는 곳에서만 생긴다.** Mesh-TensorFlow(Shazeer et al., NeurIPS 2018)가 텐서 차원에 이름을 붙이고, 장치들을 여러 축의 메시로 보고, "어떤 차원을 어떤 메시 축에 나눌지"라는 레이아웃 한 줄로 데이터 병렬과 모델 병렬을 함께 적는 방법을 브라우저에서 따라가는 인터랙티브 교육 자료입니다.

Noam Shazeer의 발표 「[Mesh-TensorFlow: Model Parallelism for Supercomputers](https://www.youtube.com/watch?v=HgGyWS40g-g)」(TF Dev Summit 2019, 15분)의 순서를 따랐고, 수식과 수치는 논문에서 가져왔습니다.

🔗 **[데모 바로가기](https://dev-jonghoonpark.github.io/how-mesh-tensorflow-works/)**

## 다루는 내용

각 절은 앞 절이 남긴 질문 하나로 끝나고, 다음 절이 그 질문을 풉니다.

1. **데이터 병렬** · 장점 세 가지와 두 개의 벽: 장치를 늘려도 평평한 장치당 메모리, 표 1의 통신/계산 비율 n/b → *왜 굳이 큰 모델이 필요한가*
2. **왜 큰 모델인가** · 발표의 링컨 문서 예시(100M vs 5B), 표 2 퍼플렉시티 35.0 → 24.0(23.5), 표 3 BLEU, 부록 C 샘플 → *모델을 어떻게 나누나*
3. **장치 배치식 모델 병렬** · GNMT식 LSTM 8층 파이프라인 타임라인, 장치 수에 비례해 커지는 그래프 → *데이터 병렬은 왜 이 문제가 없나*
4. **SPMD와 집합 통신** · 나뉜 차원을 더할 때만 all-reduce가 생기는 것을 실제 숫자로 → *batch 말고 다른 차원은?*
5. **메시와 이름 붙은 차원** · 물리 장치 8개를 [8], [2,4], [4,2], [2,2,2]로 보며 메시 축 그룹과 텐서 조각 확인
6. **두 층 MLP를 b · h · d로 나누기** · 장치별로 실제 순전파·역전파를 계산하고 단일 장치 결과와 비교하는 실행기, all-reduce 위치와 장치당 값(2d<sub>io</sub>d<sub>h</sub>, 2bd<sub>io</sub>, 2bd<sub>h</sub>)
7. **2D 메시와 분할된 all-reduce** · batch → rows, hidden → cols, 3D(io → planes), 그룹별 all-reduce
8. **레이아웃 일반 규칙과 금지 규칙** · 연산별 통신 규칙(reshape의 all-gather / all-to-all 포함), 금지·나누어떨어짐·비효율 판정기
9. **Transformer 레이아웃** · vocab · d_ff · heads 판정기, 표 2를 층 6개로 재현하는 장치당 파라미터 · 계산 · 통신 계산기(4.9B · 16 × 32)
10. **좋은 레이아웃 고르기** · 표 1 비용식, 같은 n대를 [n]으로 볼지 [r, c]로 볼지 비교, 부록 B의 2D vs 3D 분할
11. **사용법 · 결과 · 한계** · 바뀌는 건 두 줄뿐인 코드, 결과, TPU 전용 SPMD와 수동 레이아웃이라는 한계, [How Parallelism Works](https://dev-jonghoonpark.github.io/how-parallelism-works/#s5)로 이어지기
12. **퀴즈** 7문항

## 참고

- Noam Shazeer (2019), *Mesh-TensorFlow: Model Parallelism for Supercomputers*, TF Dev Summit '19 — [YouTube](https://www.youtube.com/watch?v=HgGyWS40g-g)
- Shazeer et al. (2018), *Mesh-TensorFlow: Deep Learning for Supercomputers*, NeurIPS 2018 — [arXiv:1811.02084](https://arxiv.org/abs/1811.02084)
- [tensorflow/mesh](https://github.com/tensorflow/mesh)

[how-ai-works](https://github.com/dev-jonghoonpark/how-ai-works) 시리즈의 일부입니다.
