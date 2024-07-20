using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Security.Cryptography;
using UnityEngine;

public class DataMaker : MonoBehaviour
{

    public PanoramaCapture panoramicCamera;
    private bool started_bool = false;

    // Start is called before the first frame update
    void Start()
    {
        panoramicCamera = GetComponentInChildren<PanoramaCapture>();

        if (panoramicCamera == null)
        {
            UnityEngine.Debug.LogError("PanoramicCameraCapture script not found on child object!");
        }

        //// select random position along x and z axis
        //Vector3 position = new Vector3(UnityEngine.Random.Range(-45f, 45f), 10f, UnityEngine.Random.Range(-45f, 45f));
        //Move(position);


        //// string of x_y_folder
        //string folder = position.x.ToString() + "_" + position.z.ToString() + "_folder/";

        //string folder_path = "/data/" + folder;

        //// set panoramicCamera folder_path to folder
        //panoramicCamera.folder_path = folder_path;

        //// make directory
        //System.IO.Directory.CreateDirectory(Application.dataPath + "/data/"+folder);

        // capture panorama

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {

            for (int j = 0; j < 1; j++)
            {
                // select random position along x and z axis
                Vector3 position = new Vector3(UnityEngine.Random.Range(-45f, 45f), 10f, UnityEngine.Random.Range(-45f, 45f));
                Move(position);


                // string of x_y_folder
                string folder = position.x.ToString() + "_" + position.z.ToString() + "_folder/";

                string folder_path = "/data/" + folder;

                // set panoramicCamera folder_path to folder
                panoramicCamera.folder_path = folder_path;

                // make directory
                System.IO.Directory.CreateDirectory(Application.dataPath + "/data/" + folder);

                //  loop 100 times
                for (int i = 0; i < 500; i++)
                {
                    // if not started, capture first panorama
                    if (!started_bool)
                    {
                        panoramicCamera.Capture();
                        started_bool = true;
                    }

                    // select random position along x and z axis
                    position = new Vector3(UnityEngine.Random.Range(-45f, 45f), 10f, UnityEngine.Random.Range(-45f, 45f));
                    Move(position);
                    panoramicCamera.Capture();
                }


            }

            //if (!started_bool)
            //{
            //    panoramicCamera.Capture();
            //    started_bool = true;
            //}

            //Vector3 position = new Vector3(UnityEngine.Random.Range(-45f, 45f), 10f, UnityEngine.Random.Range(-45f, 45f));
            //Move(position);
            //panoramicCamera.Capture();
        }
    }
    void Move(Vector3 position)
    {
        //Vector3 movement = new Vector3(1f, 0f, 0f);
        //transform.Translate(movement);
        transform.position = position;

    }
}
