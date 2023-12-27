from clearml import Task

def get_experiments(project_name='portfolioproject', task_name='testing api'):
    task = Task.init(project_name=project_name,  task_name=task_name)
    tasks = Task.get_tasks(project_name=project_name)
    experiment_names = [task.name for task in tasks]
    task.close()
    return experiment_names

def delete_experiments(experiments_to_delete, project_name='portfolioproject', task_name="deleteapitest"):
    task = Task.init(project_name=project_name, task_name=task_name)
    tasks = Task.get_tasks(project_name=project_name)
    status = []
    for task in tasks:
        if task.id in experiments_to_delete:
            print(task.id)
            # try:
            #     task.delete()
            #     status.append(200)
            # except Exception as e:
            #     status.append(404)
            #     continue
    task.close()
    return status

print(delete_experiments(experiments_to_delete=['test_ui']))