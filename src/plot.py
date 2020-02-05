import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image


class Plot(object):
    """
    A class for plottings
    """

    def __init__(self):
        self.df = None

    def plotWordCloud(self, ax, bow, title='Word Cloud'):
        # ax.figure(figsize=(20, 10))
        wc = WordCloud(background_color="white",
                       width=1000,
                       height=1000,
                       max_words=100,
                       relative_scaling=0.5,
                       normalize_plurals=False).generate_from_frequencies(bow)
        ax.grid(False)
        ax.set_title(title, size=36)
        ax.axis('off')
        ax.imshow(wc)

    def calculate_threshold_values(self, prob, y):
        """
        Build dataframe of the various confusion-matrix ratios by threshold
        from a list of predicted probabilities and actual y values
        """
        df = pd.DataFrame({'prob': prob, 'y': y})
        df.sort_values('prob', inplace=True)

        actual_p = df.y.sum()
        actual_n = df.shape[0] - df.y.sum()

        df['tn'] = (df.y == 0).cumsum()
        df['fn'] = df.y.cumsum()
        df['fp'] = actual_n - df.tn
        df['tp'] = actual_p - df.fn

        df['fpr'] = df.fp / (df.fp + df.tn)
        df['tpr'] = df.tp / (df.tp + df.fn)
        df['precision'] = df.tp / (df.tp + df.fp)
        df = df.reset_index(drop=True)
        return df

    def plot_roc(self, ax, prob, y, label='ROC'):
        """
        Plot roc curve for
        """
        self.df = self.calculate_threshold_values(prob, y)
        ax.plot([1] + list(self.df.fpr), [1] + list(self.df.tpr), label=label)
        x = [1] + list(self.df.fpr)
        y1 = [1] + list(self.df.tpr)
        y2 = x
        ax.fill_between(x, y1, y2, alpha=0.2)
        ax.set_xlabel('fpr')
        ax.set_ylabel('tpr')
        ax.set_title('ROC Curve')
        ax.legend()

    def plot_precision_recall(self, ax, label='precision/recall'):
        """
        Plot precision vs recall
        """
        ax.plot(self.df.tpr, self.df.precision, label=label)
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_title('Precision/Recall Curve')
        ax.set_xlim(xmin=0, xmax=1)
        ax.set_ylim(ymin=0, ymax=1)