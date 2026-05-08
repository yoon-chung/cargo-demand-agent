---
title: "External Drivers Handbook: 수요예측 외생변수 활용 가이드"
report_type: methodology
domain: air_cargo, sea_container
period: "2020-2024 데이터 기반"
route: "global, applicable to all routes"
author: "Cargo RM Team"
last_updated: "2024-11-20"
data_sources:
  - Federal Reserve FRED Database
  - ISM Manufacturing PMI
  - Freightos Baltic Index (FBX)
  - TAC Index (BAI00)
  - U.S. Energy Information Administration (EIA)
  - NY Fed Global Supply Chain Pressure Index (GSCPI)
tags: [methodology, external_drivers, fbx, pmi, oil_price, exchange_rate, gscpi, modeling]
---

# External Drivers Handbook: 수요예측 외생변수 활용 가이드

본 문서는 화물 수요예측 모델링 시 활용하는 주요 외생변수(exogenous
variables)의 정의, 데이터 출처, 수요와의 관계, 모델 input으로의 활용
가이드를 제공한다. 본 핸드북은 LightGBM, TFT 등 multi-horizon 예측 모델의
feature engineering 시 참조 문서로 사용된다.

## 1. Freight Rate Indices

### 1.1 FBX (Freightos Baltic Index)
- **정의**: 12개 글로벌 컨테이너 노선의 spot rate 동행지수, $/FEU 단위
- **출처**: Freightos (https://fbx.freightos.com), 매일 발표
- **수요와의 관계**: **수요-가격 동시 결정 변수**. FBX 상승은 캐파 부족
  (수요 강세) 신호로 해석. 단순 lag로 수요 예측에 활용 시 endogeneity 주의.
- **5년 변동 폭**: $1,300 ~ $20,000/FEU (COVID 정점)
- **모델 활용**: 가격 elastic 화주 비중이 높은 spot 시장 예측 시 -lag 변수로
  사용. Contract 시장은 영향 약함.

### 1.2 TAC Index (BAI00)
- **정의**: Baltic Air Freight Index, 글로벌 항공 화물 spot rate, $/kg
- **출처**: TAC Index (TAC), 매주 월요일 발표
- **수요와의 관계**: FBX와 유사. 공급 충격(modal shift) 및 수요 충격 동시 반영.
- **모델 활용**: 항공 수요 예측에서 z-score 표준화 후 사용

### 1.3 SCFI / CCFI / WCI (해상 추가 지수)
- SCFI: 상하이 출발 spot 지수 (가장 즉각적)
- CCFI: 중국 종합 지수
- WCI: Drewry World Container Index

## 2. Energy Prices

### 2.1 Brent Crude Oil
- **정의**: 북해 Brent유 현물 가격, $/barrel
- **출처**: EIA, FRED 시리즈 `DCOILBRENTEU`
- **수요와의 관계**: 약한 음(-)의 관계. 유가 상승 시 BAF/FSC(Fuel Surcharge)
  상승 → 운임 부담 → 일부 수요 위축.
- **5년 변동 폭**: $20 (COVID 저점) ~ $130 (러시아 침공 직후)
- **모델 활용**: monthly average로 사용. 항공은 더 민감 (jet fuel 비중 높음).

### 2.2 Jet Fuel (Kerosene-Type)
- **출처**: EIA `EER_EPJK_PF4_RGC_DPG`
- **수요와의 관계**: Brent와 +0.95 상관. 항공 화물 단위 수익성에 직접 영향.
- **모델 활용**: 항공 노선 yield 예측 모델의 핵심 변수

### 2.3 U.S. Diesel Price
- **출처**: EIA `GASDESW`
- **수요와의 관계**: 컨테이너 트럭 운송 비용. 미국 내륙 intermodal 비용에 영향
  → 일부 화주의 modal selection에 변수.

## 3. Macroeconomic Indicators

### 3.1 ISM Manufacturing PMI (US)
- **정의**: 미국 제조업 구매관리자지수, 50 기준선
- **출처**: ISM 매월 첫 영업일 발표
- **수요와의 관계**: **가장 강한 선행지표**. 2~3개월 lead. PMI > 50일 때
  미국 제조 수요 확장 → 아시아발 수입 화물 증가
- **5년 변동 폭**: 41.6 (COVID 저점) ~ 63.7 (2021 정점)
- **모델 활용**: 1~3개월 lag 변수로 사용. **가장 신뢰도 높은 외생변수.**

### 3.2 China PMI (Manufacturing, Caixin/NBS)
- **정의**: 중국 제조업 PMI. NBS(공식)와 Caixin(민간) 두 종류
- **출처**: NBS 매월 말, Caixin 매월 초
- **수요와의 관계**: 중국 수출 모멘텀의 선행지표. 1~2개월 lead.
- **모델 활용**: 중국발 수출 비중 높은 노선에서 핵심 변수

### 3.3 Industrial Production Index
- **출처**: FRED `INDPRO`
- **수요와의 관계**: PMI와 보완적. 실제 생산량 기반.

## 4. Currency

### 4.1 USD/CNY Exchange Rate
- **출처**: FRED `DEXCHUS`
- **수요와의 관계**: 위안화 약세 = 중국 수출 단가 하락 → 미국 수입 증가
  (수입 수요 자극). 1~2개월 lag 영향.
- **5년 변동 폭**: 6.30 ~ 7.35
- **모델 활용**: monthly average 사용

### 4.2 USD/KRW
- **출처**: FRED `DEXKOUS`
- **수요와의 관계**: 한국 수출 단가에 영향. ICN 출발 노선에 활용.

## 5. Supply Chain Pressure

### 5.1 GSCPI (Global Supply Chain Pressure Index)
- **정의**: NY Fed에서 발표하는 글로벌 공급망 압력 종합지수, 표준편차 단위
- **출처**: New York Fed (https://www.newyorkfed.org/research/policy/gscpi)
- **수요와의 관계**: 양(+)의 관계. GSCPI 상승은 공급망 적체 → 항공 modal
  shift 증가 → 항공 화물 수요 자극
- **5년 변동 폭**: -1.5 ~ +4.3 (COVID 정점)
- **모델 활용**: 항공 vs 해상 modal shift 모델링의 핵심 변수

### 5.2 Port Congestion (LA/LB)
- **정의**: LA/LB 항만 anchor 대기 선박 수
- **출처**: Marine Exchange of Southern California
- **수요와의 관계**: 적체 = 해상 → 항공 modal shift 신호

## 6. Industry-Specific Drivers

### 6.1 미국 소매 재고/판매 비율 (Inventory-to-Sales Ratio)
- **출처**: FRED `RETAILIRSA`
- **수요와의 관계**: 재고 비율 상승 → 추가 발주 둔화 → 수입 감소

### 6.2 반도체 가격 지수 (DRAM/HBM)
- **출처**: TrendForce, DRAMeXchange
- **수요와의 관계**: ICN 출발 화물의 baseline 결정

### 6.3 미국 holiday season retail sales
- **출처**: NRF (National Retail Federation), Census Bureau
- **수요와의 관계**: Q4 수요 강도와 직결

## 7. 변수별 Lead/Lag 종합표

| 변수 | Lead/Lag | 상관계수 (TPEB 수요 기준) | 신뢰도 |
|---|---|---|---|
| US PMI | 2~3개월 lead | +0.62 | ★★★★★ |
| China PMI | 1~2개월 lead | +0.55 | ★★★★ |
| GSCPI | 0~1개월 lead | +0.58 | ★★★★ |
| FBX | 동행 | +0.45 | ★★★ |
| USD/CNY | 1~2개월 lag | +0.40 | ★★★ |
| Brent oil | 동행 | -0.30 | ★★ |
| Inventory-to-Sales | 1~2개월 lag | -0.45 | ★★★ |

## 8. 모델 input 권장 가이드

### 8.1 LightGBM 권장 feature set
- TEU lag (t-1, t-2, ..., t-12): 12개
- US PMI (t-2): 1개
- China PMI (t-1): 1개
- GSCPI (t): 1개
- FBX (t): 1개
- Brent oil (t): 1개
- USD/CNY (t-1): 1개
- Peak season dummy (Q4 = 1): 1개
- → **총 18개 feature** (cy의 POLA 프로젝트 17개 + USD/CNY 추가 권장)

### 8.2 TFT 권장 covariate 분류
- **Static (변하지 않음)**: route, port category
- **Known future (미래 알려진 값)**: month, peak_season_dummy, holiday_dummy
- **Observed past (과거만 관찰)**: TEU history, FBX, oil, PMI, GSCPI, FX

### 8.3 백테스트 주의사항
- 외생변수의 t+h 시점 값을 백테스트에서 사용 시: 실제 운영에서는 알 수 없으므로
  주의. 권장 방법:
  - (a) 외생변수도 별도 예측 (ARIMA로 short-term 예측)
  - (b) 외생변수를 시나리오 입력으로 처리 (낙관/중립/비관)
  - (c) 직접 multi-step (각 horizon별 별도 모델, lag만 사용)

## 9. 데이터 갱신 주기

| 변수 | 갱신 주기 | API/소스 |
|---|---|---|
| FBX | 매일 | Freightos |
| TAC Index | 매주 월 | TAC API |
| US PMI | 매월 첫 영업일 | ISM |
| China PMI | 매월 1일 (NBS) | NBS / Caixin |
| GSCPI | 매월 첫째 주 | NY Fed |
| Brent oil | 매일 | EIA, FRED |
| USD/CNY | 매일 | FRED |

→ **월 단위 모델 운영 시 매월 5일 전후로 모든 외생변수 갱신 가능**

---

*본 핸드북은 FRED, ISM, EIA, NY Fed 등 공개 기관 자료를 종합한 합성 가이드
문서이며, 특정 항공사·선사의 실제 가격 민감도 데이터는 포함하지 않는다.*
