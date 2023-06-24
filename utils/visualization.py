from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from plotly import subplots
import plotly.graph_objects as go

#Adapted from https://gist.github.com/iwatobipen/f8b0e8ea2c872e7ccf34ab472454ce6c#file-chemicalspace_lapjv-ipynb

class Visualization:

    def generate_images(self, smiles):
        import base64
        from rdkit.Chem.Draw import rdMolDraw2D

        path_list = []

        mol = [Chem.MolFromSmiles(s) for s in smiles]
        c = 0
        for m in mol:
            print(f"Encoding molecule {c}: {smiles[c]}")
            drawer = rdMolDraw2D.MolDraw2DCairo(500,500)
            drawer.SetFontSize(6)
            drawer.DrawMolecule(m)
            drawer.FinishDrawing()
            text = drawer.GetDrawingText()
            imtext = base64.b64encode(text).decode('utf8')
            im_url = "data:image/png;base64, " + imtext
            path_list.append(im_url)
            c += 1

        return path_list

    def chemical_space(self, smiles, name, activity):

        # To name variables
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        sampleidx = list(range(len(mols)))
        samplemols = [mols[i] for i in sampleidx]

        # Fingerprint used: lfcfp6
        fps  = [AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=16384) for m in samplemols]

        # To define array
        def fp2arr(fp):
            arr = np.zeros((0,))
            DataStructs.ConvertToNumpyArray(fp,arr)
            return arr

        X = np.asarray([fp2arr(fp) for fp in fps])

        # To calculate PCA and TSNE
        data = PCA(n_components=94).fit_transform(X.astype(np.float32))
        embeddings = TSNE(init='pca', random_state=794, verbose=2).fit_transform(data)
        embeddings -= embeddings.min(axis=0)
        embeddings /= embeddings.max(axis=0)

        return embeddings

    def plot(self, dataset, pca=False):

        from jupyter_dash import JupyterDash
        from dash import dcc, html, Input, Output, no_update
        import plotly.graph_objects as go

        path_list = self.generate_images(dataset.smiles)
        dataset["path"] = path_list
        if pca == True:
            embeddings = self.chemical_space(dataset.smiles, dataset.id, dataset.prediction_reg)
            data_x = embeddings[:,0]
            data_y = embeddings[:,1]
        else:
            data_x = dataset.target
            data_y = dataset.target

        fig = go.Figure(data=[
            go.Scatter(
                x=data_x,
                y=data_y,
                mode="markers",
                marker=dict(
                    colorscale='viridis',
                    color=dataset.target,
                    #size=dataset.prediction_reg,
                    colorbar={"title": "Molecular<br>Weight"},
                    line={"color": "#444"},
                    reversescale=True,
                    sizeref=20,
                    sizemode="diameter",
                    opacity=0.8,
                )
            )
        ])

        # turn off native plotly.js hover effects - make sure to use
        # hoverinfo="none" rather than "skip" which also halts events.
        fig.update_traces(hoverinfo="none", hovertemplate=None, marker_size=20)

        fig.update_layout(
            xaxis=dict(title='chemical similarity'),
            yaxis=dict(title='chemical similarity'),
            plot_bgcolor='rgba(255,255,255,0.1)',
            width=800, height=800
        )

        app = JupyterDash(__name__)

        app.layout = html.Div([
            dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip"),
        ])

        @app.callback(
            Output("graph-tooltip", "show"),
            Output("graph-tooltip", "bbox"),
            Output("graph-tooltip", "children"),
            Input("graph", "hoverData"),
        )
        def display_hover(hoverData):
            if hoverData is None:
                return False, no_update, no_update

            # demo only shows the first point, but other points may also be available
            pt = hoverData["points"][0]
            bbox = pt["bbox"]
            num = pt["pointNumber"]

            df_row = dataset.iloc[num]
            img_src = df_row.path
            #name = df_row.id
            form = df_row.smiles
            #pred= df_row.prediction_reg
            # if len(desc) > 300: desc = desc[:100] + '...'

            children = [
                html.Div(children=[
                    html.Img(src=img_src, style={"width": "100%"}),
                    #html.H2(f"{name}", style={"color": "darkblue"}),
                    html.P(f"{form}"),
                    #html.P(f"{pred}"),
                ],
                style={'width': '300px', 'white-space': 'normal'})
            ]

            return True, bbox, children
        
        app.run_server(debug=True, mode='inline')

    def features_plot(self, x_data, y_data, column):
        df = x_data.join(y_data, lsuffix="_x", rsuffix="_y")
        fig, axes = plt.subplots(1,len(df.columns.values)-1, sharey=True, figsize=(15,2))

        for i, col in enumerate(df.columns.values[:-1]):
            df.plot(x=[col], y=[column], kind="scatter", ax=axes[i])

        plt.show()

    def create_BoxHist(self, df):
        # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})    

        # assigning a graph to each ax
        sns.boxplot(x=df, ax=ax_box)
        sns.histplot(data=df, ax=ax_hist, bins=100)
        ax_hist.set_yscale('log')

        # Remove x axis name for the boxplot
        ax_box.set(xlabel='')
        plt.show()

        

    def create_ComparePlot(self, results):
        results_dict = {'Original':results.T.loc['Original'],
                        'Outlier treatment':results.T.loc['Outlier treatment'],
                        'Feature redution':results.T.loc['Feature redution'],
                        'Outlier and feature redution':results.T.loc['Outlier and feature redution']}

        fig = subplots.make_subplots(rows=1, cols=2)

        for k, rows in zip(results_dict.keys(), results_dict.values()):
            for i, column in enumerate(rows):
                if i == 0:
                    pass
                else:
                    fig.add_trace(
                        go.Scatter(
                            x = rows.index,
                            y = rows[column],
                            name = column
                        ), 1,1
                    )

        for rows in results_dict.values():
            for i, column in enumerate(rows):
                if i == 0:
                    fig.add_trace(
                        go.Scatter(
                            x = rows.index,
                            y = rows[column],
                            name = column
                        ), 1,2
                    )

        fig.update_layout(
            updatemenus=[go.layout.Updatemenu(
                active=0,
                buttons=list(
                    [dict(label = 'Original',
                        method = 'update',
                        args = [{'visible': [True, True, False, False, False, False, False, False, True, False, False, False]},
                                {'title': 'Original',
                                'showlegend':True}]),
                    dict(label = 'Outlier treatment',
                        method = 'update',
                        args = [{'visible': [False, False, True, True, False, False, False, False, False, True, False, False]}, # the index of True aligns with the indices of plot traces
                                {'title': 'Outlier treatment',
                                'showlegend':True}]),
                    dict(label = 'Feature redution',
                        method = 'update',
                        args = [{'visible': [False, False, False, False, True, True, False, False, False, False, True, False]}, # the index of True aligns with the indices of plot traces
                                {'title': 'Feature redution',
                                'showlegend':True}]),
                    dict(label = 'Outlier and feature redution',
                        method = 'update',
                        args = [{'visible': [False, False, False, False, False, False, True, True, False, False, False, True]}, # the index of True aligns with the indices of plot traces
                                {'title': 'Outlier and feature redution',
                                'showlegend':True}]),
                    ])
                )
            ])

        fig.show()