from clearml import Task

def get_experiments(project_name = 'portfolioproject'):
    task = Task.init(project_name=project_name)
    tasks = Task.get_tasks(project_name=project_name)
    experiment_names = [task.name for task in tasks]
    task.close()
    return experiment_names


# print(get_experiments())