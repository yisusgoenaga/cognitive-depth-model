using System.Collections;
using UnityEngine;

public class TaskLoader : MonoBehaviour
{
    [Header("Referencias")]
    public AppManager appManager;

    private TaskData currentTask;

    public void LoadTask(TaskData task)
    {
        currentTask = task;
        Debug.Log($"Tarea {task.numero_tarea}/{task.total} | {task.dataset} | GT pendiente");
    }

    public TaskData GetCurrentTask()
    {
        return currentTask;
    }
}