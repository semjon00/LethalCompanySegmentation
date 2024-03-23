using HarmonyLib;
using UnityEngine;

namespace ScrapSegmentationGenerator.Patches
{
    [HarmonyPatch(typeof(IngamePlayerSettings))]
    internal class FPSCapMod
    {
        [HarmonyPatch("SetFramerateCap")]
        [HarmonyPostfix]
        static void SetFramerateCap(ref IngamePlayerSettings __instance)
        {
            // Extra small FPS for extra slow computers
            if (__instance.unsavedSettings.framerateCapIndex == 2)
            {
                Application.targetFrameRate = 15;
            }
        }
    }
}
