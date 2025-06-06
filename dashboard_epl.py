import streamlit as st
import pandas as pd
import numpy as np

# ============================
# PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="EPL Actual vs Prediction", 
    layout="wide",
    page_icon="🏆"
)

# Custom CSS untuk styling yang lebih bagus
st.markdown("""
<style>
    /* Main app background - PENGATURAN WARNA BACKGROUND UTAMA */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar background */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area background */
    .main .block-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling - EPL color matching */
    .main-header {
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        font-size: 2.5rem;
        font-weight: bold;
        color: #37003c !important;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    
    /* Sub headers styling - EPL purple */
    .section-header {
        color: #37003c;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    /* Sub headers styling for standings */
    .standings-header {
        color: #4682B4;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Stats cards styling - streamlit background matching */
    .stat-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: #37003c;
        border: 2px solid #37003c;
        margin: 0.5rem;
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        color: #37003c;
    }
    
    .stat-label {
        font-size: 1.3rem;
        opacity: 0.8;
        margin: 0;
        color: #37003c;
        font-weight: bold;
    }
    
    /* Team selector styling */
    .team-selector-text {
        color: #37003c;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Team selector label styling */
    .team-selector-label {
        color: #000000 !important;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Match result styling - streamlit background matching */
    .match-result {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #37003c;
    }
    
    .match-result h4 {
        color: #37003c !important;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .match-result p strong {
        color: #37003c !important;
    }
    
    /* Purple themed containers */
    .purple-container {
        background: #d2dbe9;
        border-radius: 15px;
        text-align: center;
        padding: 1.0rem;
        margin: 1rem 0;
        color: black;
    }
    
    /* No match container - matching stats cards */
    .no-match-container {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid #37003c;
        color: #37003c;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .vs-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        margin: 1rem 0;
    }
    
    /* Form styling */
    .form-result {
        display: inline-block;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        text-align: center;
        line-height: 35px;
        margin: 0 3px;
        font-weight: bold;
        color: white;
    }
    
    .win { background-color: #28a745; }
    .loss { background-color: #dc3545; }
    
    /* Highlight text styling */
    .highlight-text {
        color: #000000;
        font-size: 1.1rem;
        margin: 1rem 0;
        text-align: left;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        color: #37003c;
        font-size: 0.9rem;
        margin-top: 3rem;
        opacity: 0.8;
    }
    
    /* Dropdown styling */
    .stSelectbox > div > div {
        background-color: #37003c;
        color: white;
    }
    
    /* Custom styling for analysis headers */
    .analysis-header {
        color: #000000 !important;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        text-align: left;
    }
    
    .h2h-header {
        color: #000000 !important;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        text-align: left;
    }
    
    /* VS section styling */
    .vs-section {
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .vs-section h3 {
        color: #000000 !important;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        color: #37003c;
    }
    
    /* Override Streamlit selectbox label color */
    .stSelectbox label {
        color: #000000 !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# HEADER DENGAN LOGO EPL
# ============================
st.markdown(""" 
<style>
.logo-container img {
    background: transparent !important;
    mix-blend-mode: multiply;
    filter: 
        contrast(1.1) 
        brightness(1.1) 
        saturate(1.1)
        drop-shadow(0 0 10px rgba(55, 0, 60, 0.3));
}

.header-with-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    margin-bottom: 3rem;
    flex-wrap: wrap;
}

.main-header {
    margin: 0;
    font-size: 2.5rem;
    font-weight: bold;
}

@media (max-width: 768px) {
    .header-with-logo {
        flex-direction: column;
        gap: 10px;
    }
    .main-header {
        font-size: 2rem;
        text-align: center;
    }
}
</style>

<div class="header-with-logo logo-container">     
    <img src="https://pngdownload.io/wp-content/uploads/2023/12/Premier-League-Logo-PNG-Iconic-English-Football-Emblem-Transparent-jpg.webp"           
         alt="EPL Logo" width="80" height="80"          
         style="object-fit: contain;">     
    <h1 class="main-header">EPL 2024/2025 Dashboard - Actual vs Prediction</h1> 
</div> 
""", unsafe_allow_html=True)

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data():
    preds = pd.read_csv("EPL_2024_2025_Binary_Predictions.csv")
    actual_df = pd.read_csv("DATA FINAL EPL 2010-2025.csv", parse_dates=['Date'])

    actual_df = actual_df[actual_df['FTR'] != 'D']
    actual_df = actual_df[(actual_df['Date'] >= '2024-08-01') & (actual_df['Date'] <= '2025-05-31')]
    
    preds.rename(columns={"Actual": "actual", "Predicted": "predicted"}, inplace=True)
    preds['Date'] = pd.to_datetime(preds['Date'])
    
    mapping_result = {'Home Win': 'H', 'Away Win': 'A', 'Draw': 'D'}
    preds['actual_code'] = preds['actual'].map(mapping_result)
    preds['predicted_code'] = preds['predicted'].map(mapping_result)
    
    return preds, actual_df

preds, actual_df = load_data()

merged = preds.merge(
    actual_df[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']],
    on=['Date', 'HomeTeam', 'AwayTeam'],
    how='left'
)

# ============================
# Fungsi Hitung Poin
# ============================
def compute_points(df, use_actual=True):
    teams = sorted(set(df['HomeTeam']).union(set(df['AwayTeam'])))
    points_dict = {team: {'Points': 0, 'GD': 0, 'GF': 0, 'GA': 0} for team in teams}

    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        hg, ag = row['FTHG'], row['FTAG']
        actual = row['FTR']
        pred = row['predicted_code']
        result = actual if use_actual else pred

        if result == 'H':
            points_dict[home]['Points'] += 3
        elif result == 'A':
            points_dict[away]['Points'] += 3

        points_dict[home]['GF'] += hg
        points_dict[home]['GA'] += ag
        points_dict[home]['GD'] += hg - ag
        points_dict[away]['GF'] += ag
        points_dict[away]['GA'] += hg
        points_dict[away]['GD'] += ag - hg

    df_out = pd.DataFrame([{'Team': team, **vals} for team, vals in points_dict.items()])
    return df_out

actual_pts = compute_points(merged, use_actual=True)
predicted_pts = compute_points(merged, use_actual=False)

leaderboard = actual_pts.merge(predicted_pts, on='Team', suffixes=('_Actual', '_Predicted'))
leaderboard['Point_Diff'] = leaderboard['Points_Actual'] - leaderboard['Points_Predicted']
leaderboard = leaderboard.sort_values(by='Points_Actual', ascending=False).reset_index(drop=True)
leaderboard.insert(0, 'Rank', leaderboard.index + 1)

# ============================
# Fungsi Logo
# ============================
def get_logo_url(team_name):
    logos = {
        "Arsenal": "https://resources.premierleague.com/premierleague/badges/70/t3.png",
        "Aston Villa": "https://resources.premierleague.com/premierleague/badges/70/t7.png",
        "Bournemouth": "https://resources.premierleague.com/premierleague/badges/70/t91.png",
        "Brentford": "https://resources.premierleague.com/premierleague/badges/70/t94.png",
        "Brighton": "https://resources.premierleague.com/premierleague/badges/70/t36.png",
        "Burnley": "https://resources.premierleague.com/premierleague/badges/70/t90.png",
        "Chelsea": "https://resources.premierleague.com/premierleague/badges/70/t8.png",
        "Crystal Palace": "https://resources.premierleague.com/premierleague/badges/70/t31.png",
        "Everton": "https://resources.premierleague.com/premierleague/badges/70/t11.png",
        "Fulham": "https://resources.premierleague.com/premierleague/badges/70/t54.png",
        "Leeds": "https://resources.premierleague.com/premierleague/badges/70/t2.png",
        "Leicester": "https://resources.premierleague.com/premierleague/badges/70/t13.png",
        "Liverpool": "https://resources.premierleague.com/premierleague/badges/70/t14.png",
        "Luton": "https://resources.premierleague.com/premierleague/badges/70/t102.png",
        "Man City": "https://resources.premierleague.com/premierleague/badges/70/t43.png",
        "Man United": "https://resources.premierleague.com/premierleague/badges/70/t1.png",
        "Newcastle": "https://resources.premierleague.com/premierleague/badges/70/t4.png",
        "Norwich": "https://resources.premierleague.com/premierleague/badges/70/t45.png",
        "Nottingham Forest": "https://resources.premierleague.com/premierleague/badges/70/t17.png",
        "Sheffield Utd": "https://resources.premierleague.com/premierleague/badges/70/t23.png",
        "Southampton": "https://resources.premierleague.com/premierleague/badges/70/t20.png",
        "Tottenham": "https://resources.premierleague.com/premierleague/badges/70/t6.png",
        "Watford": "https://resources.premierleague.com/premierleague/badges/70/t57.png",
        "West Brom": "https://resources.premierleague.com/premierleague/badges/70/t35.png",
        "West Ham": "https://resources.premierleague.com/premierleague/badges/70/t21.png",
        "Wolves": "https://resources.premierleague.com/premierleague/badges/70/t39.png",
        "Ipswich": "https://resources.premierleague.com/premierleague/badges/70/t40.png",
    }
    return logos.get(team_name, "/api/placeholder/40/40")

def display_team_with_logo(team_name, size=40):
    logo_url = get_logo_url(team_name)
    return f"<img src='{logo_url}' width='{size}' style='vertical-align:middle; margin-right:8px;' onerror=\"this.src='/api/placeholder/{size}/{size}'\"> {team_name}"

# ============================
# LEADERBOARD SECTION - FIRST
# ============================
st.markdown('<div class="section-header">🏆 LEADERBOARDS</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

def style_leaderboard_purple(df):
    """Style dataframe with purple theme"""
    styled = df.style
    # Apply purple background to header and alternate row colors
    styled = styled.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#37003c'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f8f5fc')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]},
        {'selector': 'tbody td', 'props': [('color', '#37003c'), ('font-weight', '500')]},
    ])
    return styled.format({
        'Points': '{:d}',
        'Goal Diff': '{:+d}',
        'Goals For': '{:d}',
        'Goals Against': '{:d}'
    })

def style_leaderboard_white(df):
    """Style dataframe with white theme"""
    return df.style.format({
        'Points': '{:d}',
        'Goal Diff': '{:+d}',
        'Goals For': '{:d}',
        'Goals Against': '{:d}'
    })

with col1:
    st.markdown('<div class="standings-header">🔥 Actual Standings</div>', unsafe_allow_html=True)
    df_actual = leaderboard[['Rank', 'Team', 'Points_Actual', 'GD_Actual', 'GF_Actual', 'GA_Actual']].copy()
    df_actual.columns = ['Rank', 'Team', 'Points', 'Goal Diff', 'Goals For', 'Goals Against']
    
    st.dataframe(
        style_leaderboard_purple(df_actual),
        use_container_width=True,
        height=600
    )

with col2:
    st.markdown('<div class="standings-header">🔮 Predicted Standings</div>', unsafe_allow_html=True)
    df_pred = leaderboard[['Rank', 'Team', 'Points_Predicted', 'GD_Predicted', 'GF_Predicted', 'GA_Predicted']].copy()
    df_pred.columns = ['Rank', 'Team', 'Points', 'Goal Diff', 'Goals For', 'Goals Against']
    
    st.dataframe(
        style_leaderboard_purple(df_pred),
        use_container_width=True,
        height=600
    )

# Highlight Tim dengan Selisih Terbesar
highlight_team = leaderboard.loc[leaderboard['Point_Diff'].abs().idxmax()]
st.markdown(f"""
<div class="highlight-text">
    <strong>⚠️ Highlight:</strong> Tim dengan selisih poin terbesar adalah <strong>{highlight_team['Team']}</strong> 
    dengan selisih <strong>{abs(highlight_team['Point_Diff'])}</strong> poin
</div>
""", unsafe_allow_html=True)

# ============================
# STATISTIK PREDIKSI (CARDS HORIZONTAL) - SECOND
# ============================
total_games = len(merged)
correct_preds = (merged['FTR'] == merged['predicted_code']).sum()
accuracy = correct_preds / total_games * 100

st.markdown('<div class="section-header">📊 Statistik Prediksi</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <p class="stat-number">{total_games}</p>
        <p class="stat-label">Total Pertandingan</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <p class="stat-number">{correct_preds}</p>
        <p class="stat-label">Prediksi Tepat</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <p class="stat-number">{accuracy:.1f}%</p>
        <p class="stat-label">Akurasi Prediksi</p>
    </div>
    """, unsafe_allow_html=True)

# ============================
# HEAD-TO-HEAD ANALYSIS - THIRD
# ============================
st.markdown('<div class="section-header">⚔️ Head-to-Head Analysis</div>', unsafe_allow_html=True)

#st.markdown('<div class="team-selector-text">🎯 Pilih Tim untuk Analisis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    selected_home = st.selectbox(
        "🏠 Pilih Tim Home",
        sorted(merged['HomeTeam'].unique()),
        key="home_team_selectbox"
    )
with col2:
    selected_away = st.selectbox(
        "✈️ Pilih Tim Away", 
        sorted(merged['AwayTeam'].unique()),
        key="away_team_selectbox"
    )

head2head = merged[(merged['HomeTeam'] == selected_home) & (merged['AwayTeam'] == selected_away)]

# Helper functions for form display
def get_form_results(team, is_home=True):
    if is_home:
        recent = merged[merged['HomeTeam'] == team].sort_values('Date', ascending=False).head(5)
        return [('W' if r == 'H' else 'L') for r in recent['FTR']]
    else:
        recent = merged[merged['AwayTeam'] == team].sort_values('Date', ascending=False).head(5)
        return [('W' if r == 'A' else 'L') for r in recent['FTR']]

def display_form(results):
    form_html = ""
    for result in results:
        class_name = "win" if result == 'W' else "loss"
        form_html += f'<span class="form-result {class_name}">{result}</span>'
    return form_html

if head2head.empty:
    st.markdown("""
    <div class="no-match-container">
        <h3>🚫 Tidak ada pertandingan</h3>
        <p>Tidak ada pertandingan antara tim tersebut di musim ini.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Team Form Analysis
    st.markdown('<div class="analysis-header">📈 Analisis Performa Tim</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_form = get_form_results(selected_home, is_home=True)
        st.markdown(f"""
        <div class="match-result">
            <h4>{display_team_with_logo(selected_home, 30)}</h4>
            <p><strong>5 Pertandingan Terakhir (HOME)</strong></p>
            <div style="margin: 1rem 0;">
                {display_form(home_form)}
            </div>
            <p style="margin: 0; color: #666;">
                <strong>Form:</strong> {home_form.count('W')}W - {home_form.count('L')}L
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        away_form = get_form_results(selected_away, is_home=False)
        st.markdown(f"""
        <div class="match-result">
            <h4>{display_team_with_logo(selected_away, 30)}</h4>
            <p><strong>5 Pertandingan Terakhir (AWAY)</strong></p>
            <div style="margin: 1rem 0;">
                {display_form(away_form)}
            </div>
            <p style="margin: 0; color: #666;">
                <strong>Form:</strong> {away_form.count('W')}W - {away_form.count('L')}L
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Head to Head Matches
    st.markdown('<div class="h2h-header">🆚 Pertemuan Langsung</div>', unsafe_allow_html=True)
    
    
    
    # Match Details
    display_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'predicted_code']
    h2h_show = head2head[display_cols].copy()
    h2h_show['Date'] = h2h_show['Date'].dt.strftime('%d/%m/%Y')
    h2h_show['Match'] = h2h_show.apply(lambda x: f"{x['FTHG']}-{x['FTAG']}", axis=1)
    h2h_show['Result_Icon'] = h2h_show['FTR'].map({'H': '🏠', 'A': '✈️'})
    h2h_show['Prediction_Icon'] = h2h_show['predicted_code'].map({'H': '🏠', 'A': '✈️'})
    
    display_df = h2h_show[['Date', 'Match', 'Result_Icon', 'Prediction_Icon']].copy()
    display_df.columns = ['Tanggal', 'Skor', 'Hasil Actual', 'Hasil Prediksi']
    
    st.dataframe(
        display_df.style.set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#37003c'), ('color', 'white'), ('font-weight', 'bold')]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f8f5fc')]},
            {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]},
            {'selector': 'tbody td', 'props': [('color', '#37003c'), ('font-weight', '500')]},
        ]), 
        use_container_width=True
    )
    
    # H2H Summary
    if len(head2head) > 0:
        st.markdown('<div class="h2h-header">📊 Ringkasan Pertemuan</div>', unsafe_allow_html=True)
        h2h_actual = compute_points(head2head, use_actual=True)
        h2h_pred = compute_points(head2head, use_actual=False)
        h2h_summary = h2h_actual.merge(h2h_pred, on='Team', suffixes=('_Actual', '_Predicted'))
        h2h_summary = h2h_summary[h2h_summary['Team'].isin([selected_home, selected_away])]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="purple-container"><strong>📊 Actual Performance</strong></div>', unsafe_allow_html=True)
            actual_summary = h2h_summary[['Team', 'Points_Actual', 'GD_Actual', 'GF_Actual', 'GA_Actual']]
            actual_summary.columns = ['Tim', 'Poin', 'Selisih Gol', 'Gol Dibuat', 'Gol Kebobolan']
            st.dataframe(
                actual_summary.style.set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#37003c'), ('color', 'white'), ('font-weight', 'bold')]},
                    {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f8f5fc')]},
                    {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]},
                    {'selector': 'tbody td', 'props': [('color', '#37003c'), ('font-weight', '500')]},
                ]), 
                use_container_width=True
            )
        
        with col2:
            st.markdown('<div class="purple-container"><strong>🔮 Predicted Performance</strong></div>', unsafe_allow_html=True)
            pred_summary = h2h_summary[['Team', 'Points_Predicted', 'GD_Predicted', 'GF_Predicted', 'GA_Predicted']]
            pred_summary.columns = ['Tim', 'Poin', 'Selisih Gol', 'Gol Dibuat', 'Gol Kebobolan']
            st.dataframe(
                pred_summary.style.set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#37003c'), ('color', 'white'), ('font-weight', 'bold')]},
                    {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f8f5fc')]},
                    {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]},
                    {'selector': 'tbody td', 'props': [('color', '#37003c'), ('font-weight', '500')]},
                ]), 
                use_container_width=True
            )

# FOOTER
# ============================
st.markdown("---")
st.markdown("""
<div class="footer-text">
    <strong>🏆 EPL 2024/2025 Analysis Dashboard</strong><br>
    Dashboard dibuat oleh <strong>Kelompok 8</strong> | Data & Prediksi EPL 2024/2025<br>
    Powered by Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
