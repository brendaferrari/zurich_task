from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
                    colorbar={"title": "target"},
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
            xaxis=dict(title='target'),
            yaxis=dict(title='target'),
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
                style={'width': '200px', 'white-space': 'normal'})
            ]

            return True, bbox, children
        
        app.run_server(debug=True, mode='inline')

    def features_plot(self, x_data, y_data):

        fig = subplots.make_subplots(rows=1, cols=6, 
                                     subplot_titles=("Fingerprint 1","Fingerprint 2","Fingerprint 3",
                                                     "Fingerprint 4","Fingerprint 5","Fingerprint 6"))
        fig.add_trace(
            go.Scatter(mode='markers', x=x_data.iloc[:,0], y=y_data, name = 'fingerprint 1'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(mode='markers', x=x_data.iloc[:,1], y=y_data, name = 'fingerprint 2'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(mode='markers', x=x_data.iloc[:,2], y=y_data, name = 'fingerprint 3'),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(mode='markers', x=x_data.iloc[:,3], y=y_data, name = 'fingerprint 4'),
            row=1, col=4
        )
        fig.add_trace(
            go.Scatter(mode='markers', x=x_data.iloc[:,4], y=y_data, name = 'fingerprint 5'),
            row=1, col=5
        )
        fig.add_trace(
            go.Scatter(mode='markers', x=x_data.iloc[:,5], y=y_data, name = 'fingerprint 6'),
            row=1, col=6
        )
        fig.update_layout(height=400, width=1200, title_text="Target x fingerprint data visualization")
        fig.show()


    def create_BoxHist(self, df):
        fig = subplots.make_subplots(rows=1, cols=3, 
                                     subplot_titles=("Outliers boxplot", "Histogram for target distribution", 
                                                     "Histogram for log transformed target distribution"))
        fig.add_trace(
            go.Box(y=df, name = 'outliers'),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(x=df, y=df, name = 'target'),
            row=1, col=2
        )

        fig.add_trace(
            go.Histogram(x=np.log(df), y=np.log(df), name = 'target_log'),
            row=1, col=3
        )

        fig.update_layout(height=600, width=1200, title_text="Distribution visualization")
        fig.show()
        

    def create_ComparePlot(self, results):
        results_dict = {'Original':results.T.loc['Original'],
                        'Outlier treatment':results.T.loc['Outlier treatment'],
                        'Feature reduction':results.T.loc['Feature reduction'],
                        'Outlier and feature reduction':results.T.loc['Outlier and feature reduction'],
                        'Outlier and log transform':results.T.loc['Outlier and log transform']}

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
                        args = [{'visible': [True, True, False, False, False, False, False, False, False, False, True, False, False, False, False]},
                                {'title': 'Original',
                                'showlegend':True}]),
                    dict(label = 'Outlier treatment',
                        method = 'update',
                        args = [{'visible': [False, False, True, True, False, False, False, False, False, False, False, True, False, False, False]}, # the index of True aligns with the indices of plot traces
                                {'title': 'Outlier treatment',
                                'showlegend':True}]),
                    dict(label = 'Feature reduction',
                        method = 'update',
                        args = [{'visible': [False, False, False, False, False, False, True, True, False, False, False, False, True, False, False]}, # the index of True aligns with the indices of plot traces
                                {'title': 'Feature reduction',
                                'showlegend':True}]),
                    dict(label = 'Outlier and feature reduction',
                        method = 'update',
                        args = [{'visible': [False, False, False, False, False, False, False, False, True, True, False, False, False, True, False]}, # the index of True aligns with the indices of plot traces
                                {'title': 'Outlier and feature reduction',
                                'showlegend':True}]),
                    dict(label = 'Outlier and log transform',
                        method = 'update',
                        args = [{'visible': [False, False, False, False, False, False, False, False, True, True, False, False, False, False, True]}, # the index of True aligns with the indices of plot traces
                                {'title': 'Outlier and log transform',
                                'showlegend':True}]),
                    ])
                )
            ])

        fig.show()