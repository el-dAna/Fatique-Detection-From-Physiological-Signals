from clearml import Model


clearml_model_list = Model.query_models(
    # Only models from `examples` project
    project_name='portfolioproject', 
    # Only models with input name
    # model_name=None,
    # # Only models with `demo` tag or models without `TF` tag
    # tags=['demo', '-TF'],
    # # If `True`, only published models
    # only_published=False,
    # # If `True`, include archived models
    # include_archived=True,
    # # Maximum number of models returned
    # max_results=5,
    # # Only models with matching metadata
    # metadata={"key":"value"}
)