"""
Sentiment Calculation:
- Sentiment is calculated using the TextBlob library, which provides a simple API to access natural language processing (NLP) tools.
- Each message is analyzed to determine its polarity and subjectivity:
  - Polarity: Measures how positive or negative a text is, ranging from -1 (very negative) to 1 (very positive).
  - Subjectivity: Measures how subjective or objective a text is, ranging from 0 (very objective) to 1 (very subjective).


Grade Level Calculation:
- The grade level is calculated using a custom implementation of the Flesch-Kincaid Grade Level formula.
- The formula estimates the grade level required to understand a text based on the following components:
  - Number of words
  - Number of sentences
  - Number of syllables
- This formula provides an approximate U.S. school grade level, ranging from 0 (easiest) to 12 and above (most difficult),
making it easier to understand the readability of the text.
"""


import pandas as pd
import json
import re
from textblob import TextBlob
import nltk


# Download necessary NLTK data
nltk.download("punkt")

# Load the CSV file
file_path = "messages_1.csv"
data = pd.read_csv(file_path)


def contains_link(message):
    """Check if a message contains a link"""
    return bool(re.search(r"http[s]?://", message))


def contains_mentions(message):
    """Check if a message contains mentions"""
    return bool(re.search(r"@\w+", message))


def calculate_sentiment(text):
    """Calculate sentiment"""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def count_syllables(word):
    """Count syllables"""
    word = word.lower()
    vowels = "aeiouy"
    syllable_count = 0
    if word[0] in vowels:
        syllable_count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllable_count += 1
    if word.endswith("e"):
        syllable_count -= 1
    if syllable_count == 0:
        syllable_count += 1
    return syllable_count


def calculate_grade_level(text):
    """Calculate the Flesch-Kincaid Grade Level"""
    try:
        blob = TextBlob(text)
        sentences = blob.sentences
        total_words = 0
        total_sentences = len(sentences)
        total_syllables = 0

        for sentence in sentences:
            words = sentence.words
            total_words += len(words)
            total_syllables += sum(count_syllables(word) for word in words)

        if total_sentences == 0 or total_words == 0:
            return None

        average_words_per_sentence = total_words / total_sentences
        average_syllables_per_word = total_syllables / total_words

        grade_level = 0.39 * average_words_per_sentence + 11.8 * average_syllables_per_word - 15.59
        return grade_level

    except Exception as e:
        print(f"Exception occurred: {e}")
        return None


def calculate_global_stats(data):
    """Calculate global stats"""
    num_posts = len(data)
    posts_by_author = data["Author"].value_counts().to_dict()
    posts_by_channel = data["Channel"].value_counts().to_dict()
    posts_by_human_vs_bot = data["AuthorIsBot"].value_counts().to_dict()
    num_links_posted = data["Content"].apply(lambda x: contains_link(str(x))).sum()
    num_mentions = data["Content"].apply(lambda x: contains_mentions(str(x))).sum()

    # Ensure "Content" is treated as a string and handle NaN values
    data["Content"] = data["Content"].fillna("").astype(str)

    # Calculate global sentiment
    sentiments = data["Content"].apply(calculate_sentiment)
    average_polarity = sentiments.apply(lambda x: x[0]).mean()
    average_subjectivity = sentiments.apply(lambda x: x[1]).mean()

    def convert_dict(d):
        return {str(k): int(v) if isinstance(v, (int, float, pd.Int64Dtype)) else v for k, v in d.items()}

    global_stats = {
        "number_of_posts": int(num_posts),
        "number_of_posts_by_author": convert_dict(posts_by_author),
        "number_of_posts_by_channel": convert_dict(posts_by_channel),
        "number_of_posts_by_human_vs_bot": {
            "human": int(posts_by_human_vs_bot.get(False, 0)),
            "bot": int(posts_by_human_vs_bot.get(True, 0))
        },
        "number_of_links_posted": int(num_links_posted),
        "number_of_mentions": int(num_mentions),
        "global_sentiment": {
            "average_polarity": float(average_polarity),
            "average_subjectivity": float(average_subjectivity)
        }
    }

    return global_stats


def calculate_channel_specific_stats(data):
    """Calculate channel-specific stats"""
    # Calculate channel-specific stats for "LetterLoops"
    letterloops_data = data[data["Channel"] == "🧩-letterloops"]
    num_letterloops_players = letterloops_data["Author"].nunique()
    num_letterloops_puzzles_solved = len(letterloops_data)  # Assuming each post is a puzzle solved

    # Calculate channel-specific stats for "Story Sharing"
    story_sharing_data = data[data["Channel"] == "📖-story-sharing"]
    num_stories_told = len(story_sharing_data)

    # Length of stories in characters and sentences
    length_in_characters = story_sharing_data["Content"].apply(len).sum()
    length_in_sentences = story_sharing_data["Content"].apply(lambda x: len(TextBlob(x).sentences)).sum()

    # Average grade level of stories
    grade_levels = story_sharing_data["Content"].apply(calculate_grade_level)
    average_grade_level = grade_levels.dropna().mean()

    channel_specific_stats = {
        "LetterLoops": {
            "number_of_players": int(num_letterloops_players),
            "number_of_puzzles_solved": int(num_letterloops_puzzles_solved)
        },
        "Story Sharing": {
            "number_of_stories_told": int(num_stories_told),
            "length_of_stories_in_characters": int(length_in_characters),
            "length_of_stories_in_sentences": int(length_in_sentences),
            "average_grade_level": float(average_grade_level)
        }
    }

    return channel_specific_stats


def calculate_individual_stats(data):
    """Calculate individual stats"""
    individual_stats = {}
    for author in data["Author"].unique():
        author_data = data[data["Author"] == author]

        sentiments_by_author = author_data["Content"].apply(calculate_sentiment)
        avg_polarity_by_author = sentiments_by_author.apply(lambda x: x[0]).mean()
        avg_subjectivity_by_author = sentiments_by_author.apply(lambda x: x[1]).mean()

        channels_posted_to = author_data["Channel"].unique().tolist()

        individual_stats[author] = {
            "channels_posted_to": channels_posted_to,
            "average_sentiment": {
                "average_polarity": float(avg_polarity_by_author),
                "average_subjectivity": float(avg_subjectivity_by_author)
            },
        }

    return individual_stats


def main(file_path):
    """Main function to combine all stats"""
    data = pd.read_csv(file_path)

    global_stats = calculate_global_stats(data)
    channel_specific_stats = calculate_channel_specific_stats(data)
    individual_stats = calculate_individual_stats(data)

    all_stats = {
        "global_stats": global_stats,
        "channel_specific_stats": channel_specific_stats,
        "individual_stats": individual_stats
    }

    stats_json = json.dumps(all_stats, indent=4)  # Output the stats as JSON

    # Save the JSON output to a file
    output_path = "stats.json"
    with open(output_path, "w") as f:
        f.write(stats_json)

    print(f"Statistics have been saved to {output_path}")


if __name__ == "__main__":
    main(file_path)
