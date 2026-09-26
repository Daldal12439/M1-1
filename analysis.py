```python
import pandas as pd
import matplotlib.pyplot as plt
import os

# ========================================
# 기본 설정
# ========================================

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터 파일 경로
file_path = "data/OBS_ASOS_DD_20260922162003.csv"

# 이미지 저장 폴더 생성
os.makedirs("images", exist_ok=True)


# ========================================
# 1. 데이터 불러오기 및 정리
# ========================================

df = pd.read_csv(
    file_path,
    encoding="cp949"
)

print("===== 원본 데이터 =====")
print(f"데이터 크기: {df.shape}")
print("\n원본 데이터 상위 5행:")
print(df.head().to_string(index=False))


# 날짜를 날짜 형식으로 변환
df["일시"] = pd.to_datetime(df["일시"])

# 기온 데이터를 숫자 형식으로 변환
temperature_columns = [
    "평균기온(°C)",
    "최저기온(°C)",
    "최고기온(°C)"
]

for column in temperature_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# 날짜순으로 정렬
df = df.sort_values("일시").reset_index(drop=True)


# ========================================
# 2. 데이터 기본 정보
# ========================================

print("\n===== 데이터 기본 정보 =====")
print(f"데이터 수: {len(df)}개")
print(f"데이터 크기: {df.shape}")
print(f"시작 날짜: {df['일시'].min().date()}")
print(f"마지막 날짜: {df['일시'].max().date()}")

print("\n===== 결측치 확인 =====")
print(
    df[temperature_columns]
    .isnull()
    .sum()
)

print("\n===== 정리 후 데이터 상위 5행 =====")
print(
    df[
        [
            "일시",
            "평균기온(°C)",
            "최저기온(°C)",
            "최고기온(°C)"
        ]
    ]
    .head()
    .to_string(index=False)
)


# ========================================
# 3. 평균기온 기본 통계
# ========================================

print("\n===== 평균기온 기본 통계 =====")

temperature_stats = df["평균기온(°C)"].describe()

print(
    temperature_stats.to_string()
)


# ========================================
# 4. IQR을 이용한 이상치 확인
# ========================================

print("\n===== IQR 이상치 확인 =====")

q1 = df["평균기온(°C)"].quantile(0.25)
q3 = df["평균기온(°C)"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[
    (df["평균기온(°C)"] < lower_bound)
    | (df["평균기온(°C)"] > upper_bound)
]

print(f"Q1: {q1:.2f}°C")
print(f"Q3: {q3:.2f}°C")
print(f"IQR: {iqr:.2f}°C")
print(f"이상치 하한: {lower_bound:.2f}°C")
print(f"이상치 상한: {upper_bound:.2f}°C")
print(f"IQR 기준 이상치 수: {len(outliers)}개")

if len(outliers) > 0:
    print("\nIQR 기준 이상치:")
    print(
        outliers[
            ["일시", "평균기온(°C)"]
        ].to_string(index=False)
    )
else:
    print("IQR 기준 이상치가 없습니다.")


# ========================================
# 5. 7일 이동평균
# ========================================

df["7일_이동평균"] = (
    df["평균기온(°C)"]
    .rolling(window=7)
    .mean()
)

print("\n===== 7일 이동평균 확인 =====")

print(
    df[
        [
            "일시",
            "평균기온(°C)",
            "7일_이동평균"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ========================================
# 6. 전일 대비 기온 변화량
# ========================================

df["전일대비변화"] = (
    df["평균기온(°C)"].diff()
)

print("\n===== 전일 대비 기온 변화 =====")

print(
    df[
        [
            "일시",
            "평균기온(°C)",
            "전일대비변화"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print(
    f"\n전일 대비 최대 상승폭: "
    f"{df['전일대비변화'].max():.1f}°C"
)

print(
    f"전일 대비 최대 하락폭: "
    f"{df['전일대비변화'].min():.1f}°C"
)


# ========================================
# 7. 연도와 월 정보 추가
# ========================================

df["연도"] = df["일시"].dt.year
df["월"] = df["일시"].dt.month


# ========================================
# 8. 주별 평균기온
# ========================================

weekly_temperature = (
    df.set_index("일시")["평균기온(°C)"]
    .resample("W")
    .mean()
    .reset_index()
)

print("\n===== 주별 평균기온 =====")

print(
    weekly_temperature
    .head(10)
    .to_string(index=False)
)

print(
    f"\n주별 평균기온 데이터 수: "
    f"{len(weekly_temperature)}개"
)


# ========================================
# 9. 월별 평균기온
# ========================================

monthly_temperature = (
    df.groupby(
        ["연도", "월"]
    )["평균기온(°C)"]
    .mean()
    .reset_index()
)

print("\n===== 월별 평균기온 =====")

print(
    monthly_temperature
    .to_string(index=False)
)


# ========================================
# 10. 월별 평균 일교차
# ========================================

df["일교차"] = (
    df["최고기온(°C)"]
    - df["최저기온(°C)"]
)

monthly_range = (
    df.groupby(
        ["연도", "월"]
    )["일교차"]
    .mean()
    .reset_index()
)

print("\n===== 월별 평균 일교차 =====")

print(
    monthly_range
    .to_string(index=False)
)


# ========================================
# 11. 연도별 평균기온
# ========================================

yearly_temperature = (
    df.groupby("연도")["평균기온(°C)"]
    .mean()
    .reset_index()
)

print("\n===== 연도별 평균기온 =====")

print(
    yearly_temperature
    .to_string(index=False)
)


# ========================================
# 12. 연도별 표준오차 및 95% 신뢰구간
# ========================================

print("\n===== 연도별 통계량 =====")

yearly_statistics = (
    df.groupby("연도")["평균기온(°C)"]
    .agg(
        ["count", "mean", "std"]
    )
    .reset_index()
)

# 표준오차
yearly_statistics["표준오차"] = (
    yearly_statistics["std"]
    / yearly_statistics["count"] ** 0.5
)

# 95% 신뢰구간
yearly_statistics["95%_CI_하한"] = (
    yearly_statistics["mean"]
    - 1.96 * yearly_statistics["표준오차"]
)

yearly_statistics["95%_CI_상한"] = (
    yearly_statistics["mean"]
    + 1.96 * yearly_statistics["표준오차"]
)

print(
    yearly_statistics.round(3)
    .to_string(index=False)
)


# ========================================
# 13. 극단적인 평균기온 확인
# ========================================

print("\n===== 극단적인 평균기온 =====")

max_temp_row = df.loc[
    df["평균기온(°C)"].idxmax()
]

min_temp_row = df.loc[
    df["평균기온(°C)"].idxmin()
]

print(
    f"최고 평균기온: "
    f"{max_temp_row['일시'].date()} / "
    f"{max_temp_row['평균기온(°C)']:.1f}°C"
)

print(
    f"최저 평균기온: "
    f"{min_temp_row['일시'].date()} / "
    f"{min_temp_row['평균기온(°C)']:.1f}°C"
)

print(
    f"최고-최저 평균기온 차이: "
    f"{max_temp_row['평균기온(°C)'] - min_temp_row['평균기온(°C)']:.1f}°C"
)


# ========================================
# 14. 연도별 평균기온 차이
# ========================================

print("\n===== 연도별 평균기온 차이 =====")

yearly_pivot = (
    yearly_temperature
    .set_index("연도")["평균기온(°C)"]
)

difference = (
    yearly_pivot[2024]
    - yearly_pivot[2023]
)

print(
    f"2023년 평균기온: "
    f"{yearly_pivot[2023]:.2f}°C"
)

print(
    f"2024년 평균기온: "
    f"{yearly_pivot[2024]:.2f}°C"
)

print(
    f"2024년 - 2023년: "
    f"{difference:.2f}°C"
)


# ========================================
# 15. 연도별 월평균기온 차이
# ========================================

print("\n===== 연도별 월평균기온 차이 =====")

monthly_pivot = monthly_temperature.pivot(
    index="월",
    columns="연도",
    values="평균기온(°C)"
)

monthly_pivot["차이(2024-2023)"] = (
    monthly_pivot[2024]
    - monthly_pivot[2023]
)

print(
    monthly_pivot
    .round(2)
    .to_string()
)


# ========================================
# 16. 월평균기온 최고 / 최저
# ========================================

print("\n===== 월평균기온 최고/최저 =====")

highest_month = monthly_temperature.loc[
    monthly_temperature["평균기온(°C)"].idxmax()
]

lowest_month = monthly_temperature.loc[
    monthly_temperature["평균기온(°C)"].idxmin()
]

print(
    f"가장 높은 월평균기온: "
    f"{int(highest_month['연도'])}년 "
    f"{int(highest_month['월'])}월 / "
    f"{highest_month['평균기온(°C)']:.2f}°C"
)

print(
    f"가장 낮은 월평균기온: "
    f"{int(lowest_month['연도'])}년 "
    f"{int(lowest_month['월'])}월 / "
    f"{lowest_month['평균기온(°C)']:.2f}°C"
)


# ========================================
# 17. 2023년 / 2024년 3월과 8월 비교
# ========================================

print("\n===== 주요 월 비교 =====")

march_2023 = monthly_pivot.loc[3, 2023]
march_2024 = monthly_pivot.loc[3, 2024]

august_2023 = monthly_pivot.loc[8, 2023]
august_2024 = monthly_pivot.loc[8, 2024]

print(
    f"3월: 2023년 {march_2023:.2f}°C / "
    f"2024년 {march_2024:.2f}°C / "
    f"차이 {march_2024 - march_2023:.2f}°C"
)

print(
    f"8월: 2023년 {august_2023:.2f}°C / "
    f"2024년 {august_2024:.2f}°C / "
    f"차이 {august_2024 - august_2023:.2f}°C"
)


# ========================================
# 18. 월별 평균 일교차 비교
# ========================================

print("\n===== 주요 월 평균 일교차 비교 =====")

range_pivot = monthly_range.pivot(
    index="월",
    columns="연도",
    values="일교차"
)

print(
    range_pivot
    .round(2)
    .to_string()
)


# ========================================
# 19. 그래프 1
# 일별 평균기온
# ========================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["일시"],
    df["평균기온(°C)"]
)

plt.title(
    "대전 일별 평균기온 변화 (2023~2024)"
)

plt.xlabel("날짜")
plt.ylabel("평균기온 (°C)")

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "images/01_daily_temperature.png",
    dpi=150
)

plt.close()


# ========================================
# 20. 그래프 2
# 평균기온 + 7일 이동평균
# ========================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["일시"],
    df["평균기온(°C)"],
    label="일별 평균기온",
    alpha=0.4
)

plt.plot(
    df["일시"],
    df["7일_이동평균"],
    label="7일 이동평균",
    linewidth=2
)

plt.title(
    "대전 평균기온과 7일 이동평균 (2023~2024)"
)

plt.xlabel("날짜")
plt.ylabel("기온 (°C)")

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "images/02_moving_average.png",
    dpi=150
)

plt.close()


# ========================================
# 21. 그래프 3
# 2023년 / 2024년 월별 평균기온 비교
# ========================================

plt.figure(figsize=(12, 6))

for year in [2023, 2024]:

    yearly_data = monthly_temperature[
        monthly_temperature["연도"] == year
    ]

    plt.plot(
        yearly_data["월"],
        yearly_data["평균기온(°C)"],
        marker="o",
        label=str(year)
    )

plt.title(
    "대전 월별 평균기온 비교 (2023 vs 2024)"
)

plt.xlabel("월")
plt.ylabel("평균기온 (°C)")

plt.xticks(
    range(1, 13)
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "images/03_monthly_temperature.png",
    dpi=150
)

plt.close()


# ========================================
# 22. 최종 완료 메시지
# ========================================

print("\n===== 시각화 완료 =====")

print(
    "images/01_daily_temperature.png"
)

print(
    "images/02_moving_average.png"
)

print(
    "images/03_monthly_temperature.png"
)

print("\n===== 분석 완료 =====")

print(
    "대전 2023~2024년 일별 기온 데이터 분석이 완료되었습니다."
)
```
