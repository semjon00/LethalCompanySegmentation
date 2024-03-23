using UnityEngine;
using System.Collections;
using System.IO;

namespace ScrapSegmentationGenerator.Patches
{
    // Simply saves the screenshots, but async! This way it is slightly faster.
    public class ScreenshotManager : MonoBehaviour
    {
        private static ScreenshotManager instance;
        static string path = string.Format("{0}/../screenshots", Application.dataPath);

        private void Awake()
        {
            if (instance != null)
            {
                Destroy(gameObject);
                return;
            }
            instance = this;
            DontDestroyOnLoad(gameObject);
        }

        public static void ScreenshotSave(Texture2D screenShot, string sess, string shot_type)
        {
            if (instance == null)
            {
                GameObject obj = new GameObject("ScreenshotManagerObject");
                instance = obj.AddComponent<ScreenshotManager>();
                if (!Directory.Exists(path))
                {
                    Directory.CreateDirectory(path);
                }
            }

            instance.StartCoroutine(instance.ScreenshotSaveCore(screenShot, sess, shot_type));
        }

        private IEnumerator ScreenshotSaveCore(Texture2D screenShot, string sess, string shot_type)
        {
            byte[] bytes = screenShot.EncodeToPNG();
            string filename = string.Format("{0}_{1}_{2}.png", sess, Time.frameCount.ToString().PadLeft(7, '0'), shot_type);
            string fullpath = string.Format("{0}/{1}", path, filename);
            File.WriteAllBytes(fullpath, bytes);
            ScrapSegmentationGenerator.mls.LogInfo(string.Format("Took screenshot {0}", filename));
            Destroy(screenShot);
            yield return null;
        }
    }
}
