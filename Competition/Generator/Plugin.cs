using BepInEx;
using BepInEx.Configuration;
using BepInEx.Logging;
using HarmonyLib;
using ScrapSegmentationGenerator.Patches;

namespace ScrapSegmentationGenerator
{
    [BepInPlugin(modGUID, modName, modVersion)]
    public class ScrapSegmentationGenerator : BaseUnityPlugin
    {
        private const string modGUID = "Semjon.MachineVision.ScrapSegmentationGenerator";
        private const string modName = "Scrap Segmentation Generator";
        private const string modVersion = "0.3.8";

        public static ConfigEntry<bool> skipBoring;
        public static ConfigEntry<bool> doRolls;
        public static ConfigEntry<bool> extraLogging;
        public static ConfigEntry<int> framesElapsed;

        private readonly Harmony harmony = new Harmony(modGUID);

        private static ScrapSegmentationGenerator Instance;

        public static ManualLogSource mls;

        void Awake()
        {
            skipBoring = Config.Bind("Config", "Skip boring frames", true, "");
            doRolls = Config.Bind("Config", "Occasionally render all layers (slow)", false, "");
            extraLogging = Config.Bind("Config", "Log A LOT of things (slow)", false, "");
            framesElapsed = Config.Bind("Config", "How often to save frames", 15, "");

            if (Instance == null)
            {
                Instance = this;
            }
            if (mls == null)
            {
                mls = BepInEx.Logging.Logger.CreateLogSource(modGUID);
                mls.LogInfo(string.Format("Hi from {0} {1}! Tested on Lethal Company build 9th Jan 2024.", modName, modVersion));
            }

            harmony.PatchAll(typeof(Bringer));
            harmony.PatchAll(typeof(FPSCapMod));
        }
    }
}
