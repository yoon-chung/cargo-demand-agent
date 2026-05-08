---
title: "내부 가격 정책 가이드라인 (Pricing Policy Guidelines)"
report_type: policy
domain: air_cargo, sea_container
period: "2024 적용"
route: "applicable to all routes"
author: "Cargo RM Team"
last_updated: "2024-10-01"
data_sources:
  - 자체 RM 정책 문서 (합성)
  - IATA Cargo Best Practices Guide
  - Drewry Sea & Air Shipper Insight
tags: [policy, pricing, yield, gri, bsa, spot, contract, approval]
---

# 내부 가격 정책 가이드라인

본 문서는 RM(Revenue Management) 의사결정 시 가격 인상/인하, 캐파 배분, 계약
형태 결정에 대한 내부 가이드라인을 제공한다. 모든 가격 결정은 본 가이드를
일관되게 따르며, 예외 사항은 권한 등급에 따른 사전 승인을 거친다.

## 1. 가격 결정 원칙

### 1.1 3대 원칙
1. **Yield 우선**: 단순 톤수 채우기보다 단위 수익성 우선
2. **Contract와 Spot 균형**: 시장 변동성 대응을 위한 포트폴리오 관리
3. **장기 화주 우선**: 안정적 baseline 화주에 캐파 우선 배정

### 1.2 가격 트랙 분류
- **Contract (BSA, 장기 계약)**: 분기/반기/연 단위 고정 가격 (혹은 index-linked)
- **Tariff (공시 가격)**: 일반 화주 대상 standard rate
- **Spot (현물)**: booking 시점 가격, 변동성 큼

## 2. 가격 인상 조건 (GRI: General Rate Increase)

### 2.1 자동 GRI 조건 (사전 승인 불필요)
다음 조건을 동시 충족 시 즉시 +5% 이내 GRI 가능:
- 직전 4주 평균 load factor ≥ 90%
- TAC Index (또는 FBX) 직전 4주 평균이 전월비 +10% 이상
- 자사 spot booking 거절률(rejection rate) ≥ 15%

### 2.2 사전 승인 필요 GRI
- **+5~10%**: RM 팀장 승인
- **+10~20%**: RM 부서장 승인
- **+20% 이상**: 본부장 + COO 공동 승인

### 2.3 GRI 발효 통보
- Contract 화주: **45일 전 사전 통보 필수** (계약 조항 기반)
- Spot 화주: 즉시 발효 가능
- BSA 갱신 시점: 별도 협상 통해 반영

## 3. 가격 인하 조건

### 3.1 자동 인하 조건
- 직전 4주 load factor < 70% **또는**
- 자사 yield가 시장 평균(TAC Index) 대비 +10% 이상 괴리

### 3.2 단계별 대응
- **Stage 1** (LF 70~75%): 신규 spot 대상 -3~5% 할인
- **Stage 2** (LF 65~70%): 기존 contract도 분기 협상에서 -5~8% 양보
- **Stage 3** (LF < 65%): 본부장 승인 후 적극 가격 인하 + 캐파 조정

## 4. BSA(Block Space Agreement) vs Spot 비율 가이드

### 4.1 표준 비율
- **항공 화물**: BSA 65% : Tariff 15% : Spot 20%
- **해상 컨테이너**: Contract 70% : Spot 30%

### 4.2 시장 국면별 조정
- **공급 부족 국면 (캐리어 우위)**: Spot 비중 +10~15%p 확대 → yield 극대화
- **공급 과잉 국면 (화주 우위)**: BSA 비중 +10%p 확대 → 안정성 확보
- **2024 Q4 권장**: 공급 부족 국면 → Spot 30~35%로 확대

## 5. Yield 관리 임계값

### 5.1 노선별 최저 yield 기준 (예시)
| 노선 | 최저 yield ($/kg) | 비고 |
|---|---|---|
| ICN-LAX | 4.50 | 이하 booking 시 RM 팀장 승인 |
| ICN-JFK | 4.20 | - |
| ICN-FRA | 5.00 | 유럽향 프리미엄 |
| ICN-DFW | 3.80 | 신규 노선, 점진 개선 중 |

### 5.2 Capacity 활용 임계값
- **Target LF**: 항공 85%, 해상 90%
- **Yield-LF Trade-off**: LF 80% 이상부터는 yield 우선 결정

## 6. 분기별 Review 프로세스

### 6.1 정기 review
- **매주 월요일**: load factor + yield + booking pace dashboard 점검
- **매월 5일**: 전월 실적 + 외생변수 갱신 + 다음달 forecast review
- **분기 첫째 주**: 분기 실적 분석 + 다음 분기 가격 정책 결정

### 6.2 의사결정 회의체
- **Weekly RM Meeting**: RM 팀 + 영업팀 (전술적 결정)
- **Monthly Pricing Committee**: RM + 영업 + 운항 + 마케팅 (전략적 결정)
- **Quarterly Business Review**: 본부장 주재, 분기 실적 + 정책 갱신

## 7. 가격 결정 시 참조 데이터 우선순위

1. **자사 booking pace**: 가장 즉각적, 가장 신뢰도 높음
2. **TAC Index / FBX**: 시장 spot 동향
3. **forecast 모델 예측값** (LightGBM, TFT 등): 1~12개월 전망
4. **외생변수 (PMI, 유가, 환율 등)**: macro 흐름 확인
5. **경쟁사 동향**: 영업팀 채널 정보
6. **사내 시즌·노선 보고서**: 정성적 맥락

## 8. 정책 예외 사항

### 8.1 전략적 예외
다음 경우 가격 가이드 외 결정 가능 (단, 본부장 승인):
- 신규 시장 개척 시 promotion rate
- 핵심 BCO 화주의 다년 계약 (loyalty 가치)
- 정부·외교 관련 화물 (정책적 필요)

### 8.2 위기 대응 예외
- COVID급 위기: 별도 위기 대응 프레임워크 발동, 본 가이드 일시 정지
- 자연재해: 인도주의 화물 우선 (가격 면제 또는 cost only)

---

*본 가이드라인은 IATA Cargo Best Practices, Drewry Shipper Insight 등
공개 자료 기반의 합성 정책 문서이며, 특정 회사의 실제 내부 정책을 그대로
반영한 것은 아니다.*
