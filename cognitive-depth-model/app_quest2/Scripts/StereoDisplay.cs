using TMPro;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class StereoDisplay : MonoBehaviour
{
    [Header("Paneles de imagen estereoscópica")]
    public Renderer leftEyeRenderer;
    public Renderer rightEyeRenderer;

    [Header("Overlays de regiones A y B")]
    public RectTransform leftEye_A;
    public RectTransform leftEye_B;
    public RectTransform rightEye_A;
    public RectTransform rightEye_B;

    [Header("UI")]
    public TextMeshProUGUI progressText;
    public TextMeshProUGUI instructionText;
    public GameObject completedPanel;
    public GameObject taskPanel;

    private Texture2D texIzq;
    private Texture2D texDer;

    public void ShowTask(TaskData task)
    {
        if (completedPanel != null) completedPanel.SetActive(false);
        if (taskPanel != null) taskPanel.SetActive(true);

        // Actualizar progreso
        if (progressText != null)
            progressText.text = $"{task.numero_tarea} / {task.total}";

        if (instructionText != null)
            instructionText.text =
                "¿El objeto A está más CERCANO o más LEJANO de usted que el objeto B?\n" +
                "Botón A = MÁS CERCANO  |  Botón B = MÁS LEJANO";

        // Cargar imágenes desde base64
        StartCoroutine(LoadImages(task));
    }

    IEnumerator LoadImages(TaskData task)
    {
        // Imagen izquierda
        if (!string.IsNullOrEmpty(task.img_izq))
        {
            string base64 = task.img_izq.Replace("data:image/png;base64,", "")
                                        .Replace("data:image/jpeg;base64,", "");
            byte[] bytes = System.Convert.FromBase64String(base64);
            if (texIzq == null) texIzq = new Texture2D(2, 2);
            texIzq.LoadImage(bytes);
            if (leftEyeRenderer != null) leftEyeRenderer.material.mainTexture = texIzq;
        }

        // Imagen derecha
        if (!string.IsNullOrEmpty(task.img_der))
        {
            string base64 = task.img_der.Replace("data:image/png;base64,", "")
                                        .Replace("data:image/jpeg;base64,", "");
            byte[] bytes = System.Convert.FromBase64String(base64);
            if (texDer == null) texDer = new Texture2D(2, 2);
            texDer.LoadImage(bytes);
            if (rightEyeRenderer != null) rightEyeRenderer.material.mainTexture = texDer;
        }

        // Actualizar bounding boxes
        UpdateBoundingBox(leftEye_A,  task.A_x, task.A_y, task.A_ancho, task.A_alto, task.img_W, task.img_H);
        UpdateBoundingBox(leftEye_B,  task.B_x, task.B_y, task.B_ancho, task.B_alto, task.img_W, task.img_H);
        UpdateBoundingBox(rightEye_A, task.A_x, task.A_y, task.A_ancho, task.A_alto, task.img_W, task.img_H);
        UpdateBoundingBox(rightEye_B, task.B_x, task.B_y, task.B_ancho, task.B_alto, task.img_W, task.img_H);

        yield return null;
    }

    void UpdateBoundingBox(RectTransform rt, float x, float y, float w, float h,
                           float imgW, float imgH)
    {
        if (rt == null) return;
        // Normalizar coordenadas a porcentaje del panel
        float normX =  x / imgW;
        float normY =  1f - (y + h) / imgH;  // invertir Y (Unity UI origen abajo-izquierda)
        float normW =  w / imgW;
        float normH =  h / imgH;

        rt.anchorMin = new Vector2(normX, normY);
        rt.anchorMax = new Vector2(normX + normW, normY + normH);
        rt.offsetMin = Vector2.zero;
        rt.offsetMax = Vector2.zero;
    }

    public void ShowCompleted()
    {
        if (taskPanel != null) taskPanel.SetActive(false);
        if (completedPanel != null) completedPanel.SetActive(true);
        Debug.Log("¡Todas las tareas completadas!");
    }
}