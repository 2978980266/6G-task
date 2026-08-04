using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using ICities;
using UnityEngine;
using ColossalFramework.Plugins;

namespace BuildingSnapshotRecorder
{
    public class ModInfo : IUserMod
    {
        public string Name
        {
            get { return "Building Snapshot Recorder"; }
        }

        public string Description
        {
            get { return "Export one-time building geometry snapshots by building id."; }
        }
    }

    public class Loading : LoadingExtensionBase
    {
        public override void OnLevelLoaded(LoadMode mode)
        {
            if (IsGameMode(mode))
            {
                SnapshotState.ExportOnce();
            }
        }

        public override void OnReleased()
        {
            SnapshotState.Reset();
        }

        public override void OnLevelUnloading()
        {
            SnapshotState.Reset();
        }

        private static bool IsGameMode(LoadMode mode)
        {
            return mode == LoadMode.NewGame ||
                   mode == LoadMode.LoadGame ||
                   mode == LoadMode.NewGameFromScenario ||
                   mode == LoadMode.UpdateScenarioFromGame;
        }
    }

    internal static class SnapshotState
    {
        private static bool s_exported;
        private static string s_modRoot;
        private static string s_inputPath;
        private static string s_outputFolderPath;

        public static void ExportOnce()
        {
            if (s_exported)
            {
                return;
            }

            s_modRoot = GetModRoot();
            s_inputPath = Path.Combine(s_modRoot, "tracked_buildings.txt");

            List<ushort> buildingIds = LoadBuildingIds();
            if (buildingIds.Count == 0)
            {
                Log("No building ids loaded. Put one building id per line in tracked_buildings.txt.");
                return;
            }

            PrepareOutputFolder();

            for (int i = 0; i < buildingIds.Count; i++)
            {
                WriteBuildingSnapshot(buildingIds[i]);
            }

            s_exported = true;

            Log("Export finished.");
            Log("Input file: " + s_inputPath);
            Log("Output folder: " + s_outputFolderPath);
            Log("Building count: " + buildingIds.Count);
        }

        public static void Reset()
        {
            s_exported = false;
        }

        private static List<ushort> LoadBuildingIds()
        {
            List<ushort> ids = new List<ushort>();

            if (!File.Exists(s_inputPath))
            {
                Log("Input file not found: " + s_inputPath);
                return ids;
            }

            string[] lines = File.ReadAllLines(s_inputPath);

            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();

                if (line.Length == 0 || line.StartsWith("#"))
                {
                    continue;
                }

                ushort id;
                if (!ushort.TryParse(line, NumberStyles.Integer, CultureInfo.InvariantCulture, out id))
                {
                    Log("Invalid building id at line " + (i + 1) + ": " + line);
                    continue;
                }

                if (id == 0)
                {
                    Log("Ignored building id 0 at line " + (i + 1));
                    continue;
                }

                if (!ids.Contains(id))
                {
                    ids.Add(id);
                }
            }

            return ids;
        }

        private static void PrepareOutputFolder()
        {
            string outputRootDir = @"D:\vscode\6G\Cities_Skylines_data\BuildingTrackLogs";

            if (!Directory.Exists(outputRootDir))
            {
                Directory.CreateDirectory(outputRootDir);
            }

            s_outputFolderPath = Path.Combine(
                outputRootDir,
                DateTime.Now.ToString("yyyyMMdd_HHmmss")
            );

            if (!Directory.Exists(s_outputFolderPath))
            {
                Directory.CreateDirectory(s_outputFolderPath);
            }
        }

        private static void WriteHeader(StreamWriter writer)
        {
            WriteCsv(
                writer,
                "recordTimeUtc",
                "simulationFrame",
                "gameTime",
                "buildingId",
                "x",
                "y",
                "z",
                "angleY",
                "widthCells",
                "lengthCells",
                "sizeX",
                "sizeY",
                "sizeZ",
                "centerOffsetX",
                "centerOffsetY",
                "centerOffsetZ",
                "minY",
                "maxY",
                "prefabName"
            );
        }

        private static void WriteBuildingSnapshot(ushort buildingId)
        {
            if (buildingId >= BuildingManager.MAX_BUILDING_COUNT)
            {
                return;
            }

            BuildingManager buildingManager = BuildingManager.instance;
            SimulationManager simulationManager = SimulationManager.instance;

            if (buildingManager == null || simulationManager == null)
            {
                return;
            }

            Building building = buildingManager.m_buildings.m_buffer[buildingId];

            if ((building.m_flags & Building.Flags.Created) == 0)
            {
                return;
            }

            BuildingInfo info = building.Info;
            if (info == null)
            {
                return;
            }

            string fileName = "building_" + buildingId.ToString(CultureInfo.InvariantCulture) + ".csv";
            string filePath = Path.Combine(s_outputFolderPath, fileName);

            Vector3 position = building.m_position;
            float angleY = building.m_angle * 57.29578f;

            int widthCells = building.Width;
            int lengthCells = building.Length;

            Vector3 size = info.m_size;
            Vector3 centerOffset = info.m_centerOffset;

            float minY = 0f;
            float maxY = 0f;
            float unusedBaseY = 0f;

            Building.SampleBuildingHeight(
                position,
                building.m_angle,
                widthCells,
                lengthCells,
                info,
                out minY,
                out maxY,
                out unusedBaseY
            );

            using (StreamWriter writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                WriteHeader(writer);

                WriteCsv(
                    writer,
                    DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
                    simulationManager.m_currentFrameIndex.ToString(CultureInfo.InvariantCulture),
                    simulationManager.m_currentGameTime.ToString("o", CultureInfo.InvariantCulture),
                    buildingId.ToString(CultureInfo.InvariantCulture),
                    FormatFloat(position.x),
                    FormatFloat(position.y),
                    FormatFloat(position.z),
                    FormatFloat(angleY),
                    widthCells.ToString(CultureInfo.InvariantCulture),
                    lengthCells.ToString(CultureInfo.InvariantCulture),
                    FormatFloat(size.x),
                    FormatFloat(size.y),
                    FormatFloat(size.z),
                    FormatFloat(centerOffset.x),
                    FormatFloat(centerOffset.y),
                    FormatFloat(centerOffset.z),
                    FormatFloat(minY),
                    FormatFloat(maxY),
                    info.name
                );
            }
        }

        private static void WriteCsv(StreamWriter writer, params string[] cells)
        {
            for (int i = 0; i < cells.Length; i++)
            {
                if (i > 0)
                {
                    writer.Write(",");
                }

                writer.Write(EscapeCsv(cells[i]));
            }

            writer.WriteLine();
        }

        private static string EscapeCsv(string value)
        {
            if (value == null)
            {
                return "";
            }

            bool quote = value.IndexOf(',') >= 0 ||
                         value.IndexOf('"') >= 0 ||
                         value.IndexOf('\n') >= 0 ||
                         value.IndexOf('\r') >= 0;

            if (!quote)
            {
                return value;
            }

            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        private static string FormatFloat(float value)
        {
            return value.ToString("0.###", CultureInfo.InvariantCulture);
        }

        private static string GetModRoot()
        {
            try
            {
                string location = Assembly.GetExecutingAssembly().Location;

                if (!string.IsNullOrEmpty(location))
                {
                    return Path.GetDirectoryName(location);
                }
            }
            catch
            {
            }

            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                @"Colossal Order\Cities_Skylines\Addons\Mods\BuildingSnapshotRecorder"
            );
        }

        private static void Log(string message)
        {
            string full = "[BuildingSnapshotRecorder] " + message;

            Debug.Log(full);

            try
            {
                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, full);
            }
            catch
            {
            }
        }
    }
}
