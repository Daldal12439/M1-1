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

df = pd.read_csv(file_path, encoding="cp949")

# 날짜를 날짜 형식으로 변환
df["일시"] = pd.to_datetime(df["일시"])

# 기온 데이터를 숫자 형식으로 변환
temperature_columns = [
    "평균기온(°C)",
    "최저기온(°C)",
    "최고기온(°C)"
]

for column in temperature_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# 날짜순으로 정렬
df = df.sort_values("일시").reset_index(drop=True)


# ========================================
# 2. 데이터 기본 정보
# ========================================

print("===== 데이터 기본 정보 =====")
print(f"데이터 수: {len(df)}개")
print(f"시작 날짜: {df['일시'].min().date()}")
print(f"마지막 날짜: {df['일시'].max().date()}")

print("\n===== 결측치 확인 =====")
print(df[temperature_columns].isnull().sum())

print("\n===== 평균기온 기본 통계 =====")
print(df["평균기온(°C)"].describe())

print("\n===== 데이터 정리 확인 =====")
print(
    df[
        [
            "일시",
            "평균기온(°C)",
            "최저기온(°C)",
            "최고기온(°C)"
        ]
    ].head()
)


# ========================================
# 3. 7일 이동평균
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
    ].head(15)
)


# ========================================
# 4. 연도와 월 정보 추가
# ========================================

df["연도"] = df["일시"].dt.year
df["월"] = df["일시"].dt.month


# ========================================
# 5. 월별 평균기온
# ========================================

monthly_temperature = (
    df.groupby(["연도", "월"])["평균기온(°C)"]
    .mean()
    .reset_index()
)

print("\n===== 월별 평균기온 =====")
print(
    monthly_temperature.to_string(index=False)
)


# ========================================
# 6. 월별 평균 일교차
# ========================================

df["일교차"] = (
    df["최고기온(°C)"] -
    df["최저기온(°C)"]
)

monthly_range = (
    df.groupby(["연도", "월"])["일교차"]
    .mean()
    .reset_index()
)

print("\n===== 월별 평균 일교차 =====")
print(
    monthly_range.to_string(index=False)
)


# ========================================
# 7. 연도별 평균기온
# ========================================

yearly_temperature = (
    df.groupby("연도")["평균기온(°C)"]
    .mean()
    .reset_index()
)

print("\n===== 연도별 평균기온 =====")
print(
    yearly_temperature.to_string(index=False)
)


# ========================================
# 8. 극단적인 평균기온 확인
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


# ========================================
# 9. 연도별 평균기온 차이
# ========================================

print("\n===== 연도별 평균기온 차이 =====")

yearly_pivot = (
    yearly_temperature
    .set_index("연도")["평균기온(°C)"]
)

difference = (
    yearly_pivot[2024] -
    yearly_pivot[2023]
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
# 10. 연도별 월평균기온 차이
# ========================================

print("\n===== 연도별 월평균기온 차이 =====")

monthly_pivot = monthly_temperature.pivot(
    index="월",
    columns="연도",
    values="평균기온(°C)"
)

monthly_pivot["차이(2024-2023)"] = (
    monthly_pivot[2024] -
    monthly_pivot[2023]
)

print(
    monthly_pivot.round(2).to_string()
)


# ========================================
# 11. 월평균기온 최고 / 최저
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
# 12. 그래프 1
# 일별 평균기온
# ========================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["일시"],
    df["평균기온(°C)"]
)

plt.title("대전 일별 평균기온 변화 (2023~2024)")
plt.xlabel("날짜")
plt.ylabel("평균기온 (°C)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "images/01_daily_temperature.png",
    dpi=150
)

plt.close()


# ========================================
# 13. 그래프 2
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
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "images/02_moving_average.png",
    dpi=150
)

plt.close()


# ========================================
# 14. 그래프 3
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
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "images/03_monthly_temperature.png",
    dpi=150
)

plt.close()


# ========================================
# 15. 최종 완료 메시지
# ========================================

print("\n===== 시각화 완료 =====")
print("images/01_daily_temperature.png")
print("images/02_moving_average.png")
print("images/03_monthly_temperature.png")

print("\n===== 분석 완료 =====")
print("대전 2023~2024년 일별 기온 데이터 분석이 완료되었습니다.")