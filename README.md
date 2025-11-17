# John Lewis 2025 Christmas Ad: NLP Audience Analysis

This project performs a comprehensive NLP analysis of public reception to the 2025 John Lewis "Where Love Lives" Christmas advert.

It analyzes over 1,200 public comments from YouTube and Reddit, using two different NLP techniques to understand audience sentiment and the key themes of discussion.

1.  **Sentiment Analysis:** A Hugging Face Transformer model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) classifies the emotion (Positive, Negative, Neutral) of each comment.
2.  **Topic Modeling:** Traditional TF-IDF + LDA techniques are used to discover the 4 main themes of discussion.

The final results are presented in a fully interactive Streamlit dashboard.

## 📊 Dashboard & Analysis

The dashboard provides a high-level overview of the ad's reception and allows for deep-diving into the data.

* **Overall Sentiment:** A pie chart shows the aggregate sentiment split across all comments.
* **Discovered Topics:** A table displays the 4 main topics discovered by the LDA model, along with their key keywords.
* **Sentiment by Topic:** A treemap visualizes the sentiment breakdown *within* each topic, allowing us to answer the core research question.
* **Data Explorer:** A filterable, searchable table of all 1,200+ comments with their predicted sentiment and topic.
* **CSV Export:** A download button to export the complete, analyzed dataset.

`![Screenshots of the Streamlit dashboard](dashboard_screenshot.png)(dashboard_screenshot1.png)`

---

## 🚀 How to Run This Project

There are two ways to run this project:

1.  **Analysis-Only (Recommended):** Use the pre-scraped data to run the dashboard.
2.  **Full Re-scrape:** Run the scraper scripts yourself to collect fresh data.

### 1. Installation

First, clone the repository and set up your environment.

```bash
git clone [https://github.com/Sarfarazzzzz/Sentiment-Analysis-on-John-Lewis-2025-Christmas-Ad.git](https://github.com/Sarfarazzzzz/Sentiment-Analysis-on-John-Lewis-2025-Christmas-Ad.git)
cd your-repo-name

# It is highly recommended to use a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all required libraries
pip install -r requirements.txt
