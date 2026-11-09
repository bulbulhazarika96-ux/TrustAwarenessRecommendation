
import pandas as pd


# Load news articles
def load_news():

    columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities"
    ]

    news = pd.read_csv(
        "dataset/train/news.tsv",
        sep="\t",
        header=None,
        names=columns
    )

    news["title"] = news["title"].fillna("")
    news["abstract"] = news["abstract"].fillna("")

    news["content"] = (
        news["title"] + " " + news["abstract"]
    )

    return news


# Load user behaviour
def load_behaviors():

    columns = [
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions"
    ]

    behaviors = pd.read_csv(
        "dataset/train/behaviors.tsv",
        sep="\t",
        header=None,
        names=columns
    )

    behaviors["history"] = behaviors["history"].fillna("")
    behaviors["impressions"] = behaviors["impressions"].fillna("")

    return behaviors


# Test both datasets
if __name__ == "__main__":

    news = load_news()
    behaviors = load_behaviors()

    print("\n===== NEWS DATA =====")

    print(news[["news_id", "category", "title"]].head())

    print("\nTotal News:", len(news))

    print("\n===== USER BEHAVIOUR DATA =====")

    print(behaviors.head())

    print("\nTotal Behaviour Records:", len(behaviors))

    print("\nTotal Unique Users:", behaviors["user_id"].nunique())
