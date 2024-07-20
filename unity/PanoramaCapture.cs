using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;

public class PanoramaCapture : MonoBehaviour
{

    public Camera targetCamera;
    public RenderTexture cubeMapLeft;
    public RenderTexture equirectRT;
    public Stopwatch stopwatch;
    public int file_counter = 0;
    public string folder_path = "/Panoramas/";

    // Start is called before the first frame update
    public void Start()
    {
        stopwatch = new Stopwatch();
    }

    // Update is called once per frame
    public void Update()
    {
        // if space key is pressed, capture the panorama
        //if (Input.GetKeyDown(KeyCode.Space))
        //{
        //    Capture();
        //}
        
    }

    public void Capture() {
        // Get image and save data
        stopwatch.Start();
        targetCamera.RenderToCubemap(cubeMapLeft);
        cubeMapLeft.ConvertToEquirect(equirectRT);
        file_counter++;
        Save(equirectRT);
        stopwatch.Stop();

        // calculate the time taken to capture the panorama
        System.TimeSpan ts = stopwatch.Elapsed;
        string elapsedTime = string.Format("{0:00}:{1:00}:{2:00}.{3:000}",
                ts.Hours, ts.Minutes, ts.Seconds, ts.Milliseconds);
        UnityEngine.Debug.Log("RunTime: " + elapsedTime);
        stopwatch.Reset();
    }

    public void Save(RenderTexture rt)
    {
        Texture2D tex = new Texture2D(rt.width, rt.height);//, TextureFormat.RGB24, false);

        RenderTexture.active = rt;

        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);

        // convert image to size 256 width 128 height
        //tex.Resize(256, 128);

        byte[] bytes = tex.EncodeToJPG();

        string file_name = "panorama_" + file_counter.ToString();
        string image_file_name = file_name + ".jpg";
        string pos_file_name = file_name + ".txt";
        string folder_name = folder_path;
        string path = Application.dataPath + folder_name + image_file_name;

        System.IO.File.WriteAllBytes(path, bytes);

        // save the camera position
        string pos = targetCamera.transform.position.ToString();
        string rot = targetCamera.transform.rotation.ToString();
        string[] lines = { pos, rot };
        System.IO.File.WriteAllLines(Application.dataPath + folder_name + pos_file_name, lines);



    }
}
