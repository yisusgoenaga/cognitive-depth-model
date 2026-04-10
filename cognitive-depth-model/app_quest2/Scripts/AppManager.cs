using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class AppManager : MonoBehaviour
{
    [Header("Configuración del servidor")]
    public string serverIP = "192.168.1.144";
    public int serverPort = 5000;
    public string participantID = "P001";

    [Header("Referencias")]
    public StereoDisplay stereoDisplay;
    public TaskLoader taskLoader;

    private string baseURL;
    private int currentTaskIndex = 0;
    private int totalTasks = 0;
    private float taskStartTime;
    private bool isResponding = false;

    void Start()
    {
        baseURL = $"http://{serverIP}:{serverPort}";
        StartCoroutine(InitSession());
    }

    IEnumerator InitSession()
    {
        string url = $"{baseURL}/api/iniciar";
        string json = $"{{\"participant_id\":\"{participantID}\"}}";
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
        {
            req.uploadHandler   = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<InitResponse>(req.downloadHandler.text);
                totalTasks = response.total;
                Debug.Log($"Sesión iniciada: {totalTasks} tareas");
                StartCoroutine(LoadTask(0));
            }
            else
            {
                Debug.LogError($"Error iniciando sesión: {req.error}");
            }
        }
    }

    IEnumerator LoadTask(int idx)
    {
        string url = $"{baseURL}/api/tarea/{idx}";
        using (UnityWebRequest req = UnityWebRequest.Get(url))
        {
            yield return req.SendWebRequest();
            if (req.result == UnityWebRequest.Result.Success)
            {
                var task = JsonUtility.FromJson<TaskData>(req.downloadHandler.text);
                if (task.fin)
                {
                    stereoDisplay.ShowCompleted();
                    yield break;
                }
                taskLoader.LoadTask(task);
                stereoDisplay.ShowTask(task);
                taskStartTime = Time.time;
                isResponding = false;
            }
        }
    }

    public void RegisterResponse(string response)
    {
        if (isResponding) return;
        isResponding = true;
        int responseTimeMs = Mathf.RoundToInt((Time.time - taskStartTime) * 1000);
        StartCoroutine(SendResponse(currentTaskIndex, response, responseTimeMs));
    }

    IEnumerator SendResponse(int idx, string response, int timeMs)
    {
        string url  = $"{baseURL}/api/respuesta";
        string json = $"{{\"idx\":{idx},\"respuesta\":\"{response}\"}}";
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
        {
            req.uploadHandler   = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return req.SendWebRequest();

            currentTaskIndex++;
            StartCoroutine(LoadTask(currentTaskIndex));
        }
    }

    void Update()
    {
        if (isResponding) return;

        // Botón A del control derecho = más cercano
        if (OVRInput.GetDown(OVRInput.Button.One))
            RegisterResponse("mas_cercano");

        // Botón B del control derecho = más lejano  
        if (OVRInput.GetDown(OVRInput.Button.Two))
            RegisterResponse("mas_lejano");

        // Botón X del control izquierdo también = más cercano
        if (OVRInput.GetDown(OVRInput.Button.Three))
            RegisterResponse("mas_cercano");

        // Botón Y del control izquierdo también = más lejano
        if (OVRInput.GetDown(OVRInput.Button.Four))
            RegisterResponse("mas_lejano");
    }
}

[System.Serializable]
public class InitResponse { public bool ok; public int total; }

[System.Serializable]
public class TaskData
{
    public int idx;
    public int total;
    public int numero_tarea;
    public string dataset;
    public string tipo_tarea;
    public string nivel_disparidad;
    public string presencia_distractores;
    public string img_izq;
    public string img_der;
    public float A_x, A_y, A_ancho, A_alto;
    public float B_x, B_y, B_ancho, B_alto;
    public float img_H, img_W;
    public bool fin;
}