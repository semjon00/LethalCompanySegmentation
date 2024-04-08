using GameNetcodeStuff;
using HarmonyLib;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;
using UnityEngine.Rendering;

namespace ScrapSegmentationGenerator.Patches
{
    [HarmonyPatch(typeof(PlayerControllerB))]
    public class Bringer : MonoBehaviour
    {
        static public int resWidth = 860; // (Vanilla resolution)
        static public int resHeight = 520;

        static int scrapLayer = 1 << 6;
        static int HUDLayer = 1 << 7;
        static int enemiesLayer = 1 << 19;
        static int emptyLayer = 1 << 30;

        static string sess = "";

        // Camera must remain still while we take the pictures.
        // Maybe not necessary, but I don't want to test if it is.
        static Vector3 camera_pos = Vector3.up;
        static Vector3 camera_rot = Vector3.up;

        static GameObject ambientLightObject; // Extra light to lighten the dark.
        static Material redMaterial; // To color all scrap and enemies red.
        static GameObject gigaCube; // Blue screen. Like green screen, but blue. Ugly.

        [HarmonyPatch("LateUpdate")]
        [HarmonyPostfix]
        static void Update(ref PlayerControllerB __instance)
        {
            if (sess.Length < 2)
            {
                System.Random random = new System.Random();
                sess = (random.Next() % 10000).ToString().PadLeft(4, '0');
            }

            if (redMaterial == null)
            {
                redMaterial = SolidColorMaterial(new Color(0.95f, 0.0f, 0.0f));
            }
            if (gigaCube == null)
            {
                gigaCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                gigaCube.layer = 30; // empty

                MeshRenderer renderer = gigaCube.GetComponent<MeshRenderer>();
                if (renderer == null) renderer = gigaCube.AddComponent<MeshRenderer>();
                renderer.material = SolidColorMaterial(new Color(0.0f, 0.0f, 900000.0f));
                renderer.receiveShadows = false;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                gigaCube.transform.localScale = new Vector3(2.0f, 0.5f, 2.0f) * 570.0f;
                gigaCube.SetActive(false);
            }

            if (ScrapSegmentationGenerator.skipBoring.Value)
            {
                if (__instance.inTerminalMenu || __instance.isTypingChat || (!__instance.AllowPlayerDeath()))
                {
                    return;
                }
            }

            if (__instance.gameplayCamera.enabled && Time.frameCount % ScrapSegmentationGenerator.framesElapsed.Value == 0)
            {
                var cam = __instance.gameplayCamera;

                // Putting up gloves
                var cam_transformparent_bk = cam.transform.parent;
                var rt_active_bk = RenderTexture.active;
                var cam_targetTexture_bk = cam.targetTexture;

                cam.transform.SetParent(null);
                FixPos(ref cam);

                LayerSave(ref cam, ~emptyLayer);

                var lights = TinkerLights();
                try
                {
                    GameObject val = GameObject.Find("Systems");
                    val.transform.Find("Rendering").Find("CustomPass").gameObject.SetActive(false);
                    val.transform.Find("Rendering").Find("VolumeMain").gameObject.SetActive(false);
                } finally { }


                if (ScrapSegmentationGenerator.doRolls.Value && Time.frameCount % 120 == 0)
                {
                    for (var i = 0; i < 32; i++)
                    {
                        LayerSave(ref cam, 1 << i);
                    }
                }

                gigaCube.SetActive(true);
                gigaCube.transform.position = camera_pos + cam.transform.forward * 600.0f;
                gigaCube.transform.rotation = cam.transform.rotation;
                LayerSave(ref cam, scrapLayer | emptyLayer);
                LayerSave(ref cam, enemiesLayer | emptyLayer);
                gigaCube.SetActive(false);
                TinkerLights2(ref lights);

                LayerSave(ref cam, ~(HUDLayer | scrapLayer | enemiesLayer));
                LayerSave(ref cam, ~HUDLayer);

                try
                {
                    GameObject val = GameObject.Find("Systems");
                    val.transform.Find("Rendering").Find("CustomPass").gameObject.SetActive(true);
                    val.transform.Find("Rendering").Find("VolumeMain").gameObject.SetActive(true);
                }
                finally { }
                RestoreLights(ref lights);

                // Cleaning up the crime scene
                cam.transform.parent = cam_transformparent_bk;
                RenderTexture.active = rt_active_bk;
                cam.targetTexture = cam_targetTexture_bk;

                if (Time.frameCount % 240 == 0 && ScrapSegmentationGenerator.extraLogging.Value)
                {
                    LayersList();
                    ShadersList();
                    CameraProperties(cam);
                    if (Time.frameCount % 1200 == 0)
                    {
                        AllObjectsEnumerate();
                    }
                }
            }
        }

        static Material SolidColorMaterial(Color c)
        {
            var m = new Material(Shader.Find("HDRP/Unlit"));
            m.color = c;
            m.DisableKeyword("_EMISSION");
            m.DisableKeyword("_SPECULARHIGHLIGHTS_OFF");
            m.DisableKeyword("_GLOSSYREFLECTIONS_OFF");
            m.SetFloat("_Brightness", 9000.0f);
            return m;
        }

        static void FixPos(ref Camera cam)
        {
            camera_pos = cam.transform.position + Vector3.zero;
            camera_rot = cam.transform.eulerAngles + Vector3.zero;
        }

        static void RestorePos(ref Camera cam)
        {
            cam.transform.position = camera_pos;
            cam.transform.eulerAngles = camera_rot;
        }

        static List<LightState> TinkerLights()
        {
            List<LightState> originalStates = new List<LightState>();
            Light[] lights = FindObjectsOfType<Light>();
            foreach (Light light in lights)
            {
                originalStates.Add(new LightState
                {
                    light = light,
                    is_enabled = light.enabled,
                    shadows = light.shadows,
                    intensity = light.intensity,
                });
            }

            foreach (Light light in lights)
            {
                light.shadows = LightShadows.None;
                light.enabled = false;
            }

            return originalStates;
        }

        static void TinkerLights2(ref List<LightState> originalStates)
        {
            foreach (var state in originalStates)
            {
                state.light.shadows = LightShadows.None;
                state.light.enabled = state.is_enabled;
                state.light.intensity /= 1.3f;
            }

            if (ambientLightObject == null)
            {
                ambientLightObject = new GameObject("Ambient Light");
                ambientLightObject.transform.position = camera_pos;
                Light ambientLight = ambientLightObject.AddComponent<Light>();
                // https://forum.unity.com/threads/culling-mask-not-working-correctly-in-urp-12-0-unity-2021-2-0b.1152764/
                // Nice, we can not use this absolutely necessary thing...
                ambientLight.type = LightType.Point;
                ambientLight.shadows = LightShadows.None;
                ambientLight.color = new Color(0.7f, 0.9f, 0.7f);
                ambientLight.range = 10000.0f;
                ambientLight.intensity = 10.0f;
            }
        }

        static void RestoreLights(ref List<LightState> originalStates)
        {
            Destroy(ambientLightObject);
            ambientLightObject = null;

            foreach (LightState state in Enumerable.Reverse(originalStates))
            {
                state.light.enabled = state.is_enabled;
                state.light.shadows = state.shadows;
                state.light.intensity = state.intensity;
            }
        }

        // Not actually re-rendering the reference picture would have been nice.
        // But it was harder, because of non-alignment. Oh well.
        // Saves picture from camera, respecting the provided culling mask.
        static void LayerSave(ref Camera cam, int cull)
        {
            var cam_cullingMask_bk = cam.cullingMask;
            cull &= cam.cullingMask | emptyLayer;
            cam.cullingMask = cull;
            RestorePos(ref cam);
            RenderTexture renderTexture = new RenderTexture(resWidth, resHeight, 32);
            cam.targetTexture = renderTexture;
            cam.Render();
            RenderTexture.active = renderTexture;
            Texture2D screenshot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
            screenshot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
            screenshot.Apply();
            ScreenshotSave(screenshot, string.Format("layer{0}", cull));
            Destroy(screenshot);
            Destroy(renderTexture);
            cam.cullingMask = cam_cullingMask_bk;
        }

        static void ScreenshotSave(Texture2D screenShot, string shot_type)
        {
            ScreenshotManager.ScreenshotSave(screenShot, sess, shot_type);
        }

        static void LayersList()
        {
            int layerCount = 32;
            string[] layerNames = new string[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                layerNames[i] = i + ":" + LayerMask.LayerToName(i);
            }
            ScrapSegmentationGenerator.mls.LogInfo(string.Format("Layers: {0}", string.Join(", ", layerNames)));
        }

        static void ShadersList()
        {
            Shader[] shaders = Resources.FindObjectsOfTypeAll<Shader>();
            string[] names = new string[shaders.Length];
            for (int i = 0; i < shaders.Length; i++)
            {
                names[i] = shaders[i].name;
            }
            ScrapSegmentationGenerator.mls.LogInfo(string.Format("Shaders: {0}", string.Join(", ", names)));
        }

        static void CameraProperties(Camera mainCamera)
        {
            if (mainCamera != null)
            {
                // Get all properties of the Camera class
                PropertyInfo[] properties = typeof(Camera).GetProperties(BindingFlags.Instance | BindingFlags.Public);
                FieldInfo[] fields = typeof(Camera).GetFields(BindingFlags.Instance | BindingFlags.Public);
                foreach (var property in properties)
                {
                    ScrapSegmentationGenerator.mls.LogInfo("prop  " + property.Name + ": " + property.GetValue(mainCamera, null));
                }
                foreach (var field in fields)
                {
                    ScrapSegmentationGenerator.mls.LogInfo("field " + field.Name + ": " + field.GetValue(mainCamera));
                }
            }
        }

        static void AllObjectsEnumerate()
        {
            void Traverse(GameObject obj, int depth=0)
            {
                ScrapSegmentationGenerator.mls.LogInfo(new string('>', depth) + " " + obj.name + " // " + obj.GetType().FullName + " // " + obj.tag + " // " + obj.layer);
                foreach (Transform child in obj.transform)
                {
                    Traverse(child.gameObject, depth + 1);
                }
            }

            foreach (GameObject obj in FindObjectsOfType(typeof(GameObject)))
            {
                if (obj.transform.parent == null)
                {
                    Traverse(obj);
                }
            }
        }
    }

    public class LightState
    {
        public Light light;
        public bool is_enabled;
        public LightShadows shadows;
        public float intensity;
    }

    public class RendererState
    {
        public Renderer renderer;
        public Material originalMaterial;
    }
}
