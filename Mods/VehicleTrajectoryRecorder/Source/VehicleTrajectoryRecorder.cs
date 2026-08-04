using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using ICities;
using UnityEngine;
using ColossalFramework.Plugins;

namespace VehicleTrajectoryRecorder
{
    public class ModInfo : IUserMod
    {
        public string Name
        {
            get { return "Vehicle Trajectory Recorder"; }
        }

        public string Description
        {
            get { return "Record vehicle trajectories by vehicle id to CSV."; }
        }
    }

    public class Loading : LoadingExtensionBase
    {
        public override void OnLevelLoaded(LoadMode mode)
        {
            if (IsGameMode(mode))
            {
                RecorderState.Start();
            }
        }

        public override void OnLevelUnloading()
        {
            RecorderState.Stop("level unloading");
        }

        public override void OnReleased()
        {
            RecorderState.Stop("released");
        }

        private static bool IsGameMode(LoadMode mode)
        {
            return mode == LoadMode.NewGame ||
                   mode == LoadMode.LoadGame ||
                   mode == LoadMode.NewGameFromScenario ||
                   mode == LoadMode.UpdateScenarioFromGame;
        }
    }

    public class Threading : ThreadingExtensionBase
    {
        public override void OnAfterSimulationTick()
        {
            RecorderState.OnAfterSimulationTick();
        }

        public override void OnReleased()
        {
            RecorderState.Stop("threading released");
        }
    }

    internal static class RecorderState
    {
        private const int SampleEveryFrames = 16;
        private const float MaxDurationSeconds = 0f; // 0 means unlimited until map unload.
        private const int FlushEveryLines = 20;

        private static readonly List<ushort> s_vehicleIds = new List<ushort>();

        private static bool s_running;
        private static uint s_lastSampleFrame;
        private static float s_sessionSeconds;
        private static int s_linesSinceFlush;

        private static string s_modRoot;
        private static string s_inputPath;
        private static string s_outputFolderPath;
        private static readonly Dictionary<ushort, StreamWriter> s_writers = new Dictionary<ushort, StreamWriter>();

        public static void Start()
        {
            Stop("restart");

            s_modRoot = GetModRoot();
            s_inputPath = Path.Combine(s_modRoot, "tracked_vehicles.txt");

            LoadVehicleIds();

            if (s_vehicleIds.Count == 0)
            {
                Log("No vehicle ids loaded. Put one vehicle id per line in tracked_vehicles.txt.");
                return;
            }

            PrepareOutputFile();

            s_running = true;
            s_lastSampleFrame = 0u;
            s_sessionSeconds = 0f;
            s_linesSinceFlush = 0;

            Log("Started. Vehicle count: " + s_vehicleIds.Count);
            Log("Input file: " + s_inputPath);
            Log("Output folder: " + s_outputFolderPath);
        }

        public static void Stop(string reason)
        {
            foreach (KeyValuePair<ushort, StreamWriter> pair in s_writers)
            {
                try
                {
                    pair.Value.Flush();
                    pair.Value.Close();
                }
                catch
                {
                }
            }

            s_writers.Clear();

            if (s_running)
            {
                Log("Stopped: " + reason);
            }

            s_running = false;
            s_lastSampleFrame = 0u;
            s_sessionSeconds = 0f;
            s_linesSinceFlush = 0;
            s_vehicleIds.Clear();
        }

        public static void OnAfterSimulationTick()
        {
            if (!s_running || s_writers.Count == 0)
            {
                return;
            }

            SimulationManager simulationManager = SimulationManager.instance;
            if (simulationManager == null)
            {
                return;
            }

            uint currentFrame = simulationManager.m_currentFrameIndex;
            if (currentFrame == 0u)
            {
                return;
            }

            float simulationTimeDelta = simulationManager.m_simulationTimeDelta;
            if (simulationTimeDelta > 0f)
            {
                s_sessionSeconds += simulationTimeDelta;
            }

            if (MaxDurationSeconds > 0f && s_sessionSeconds > MaxDurationSeconds)
            {
                Stop("max duration reached");
                return;
            }

            if (currentFrame == s_lastSampleFrame)
            {
                return;
            }

            s_lastSampleFrame = currentFrame;

            if ((currentFrame % SampleEveryFrames) != 0u)
            {
                return;
            }

            SampleAllVehicles();
        }

        private static void LoadVehicleIds()
        {
            s_vehicleIds.Clear();

            if (!File.Exists(s_inputPath))
            {
                Log("Input file not found: " + s_inputPath);
                return;
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
                    Log("Invalid vehicle id at line " + (i + 1) + ": " + line);
                    continue;
                }

                if (id == 0)
                {
                    Log("Ignored vehicle id 0 at line " + (i + 1));
                    continue;
                }

                if (!s_vehicleIds.Contains(id))
                {
                    s_vehicleIds.Add(id);
                }
            }
        }

        private static void PrepareOutputFile()
        {
            string outputRootDir = @"D:\vscode\6G\Cities_Skylines_data\VehicleTrackLogs";

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

            s_writers.Clear();

            for (int i = 0; i < s_vehicleIds.Count; i++)
            {
                ushort vehicleId = s_vehicleIds[i];
                string fileName = "vehicle_" + vehicleId.ToString(CultureInfo.InvariantCulture) + ".csv";
                string filePath = Path.Combine(s_outputFolderPath, fileName);

                StreamWriter writer = new StreamWriter(filePath, false, Encoding.UTF8);
                writer.AutoFlush = false;
                s_writers[vehicleId] = writer;

                WriteHeader(writer);
            }
        }

        private static void WriteHeader(StreamWriter writer)
        {
            WriteCsv(
                writer,
                "recordTimeUtc",
                "sessionSeconds",
                "simulationFrame",
                "gameTime",
                "vehicleId",
                "x",
                "y",
                "z",
                "angleX",
                "angleY",
                "speed",
                "prefabName"
            );
        }

        private static void SampleAllVehicles()
        {
            VehicleManager vehicleManager = VehicleManager.instance;
            SimulationManager simulationManager = SimulationManager.instance;

            uint simulationFrame = 0;
            string gameTime = "";

            if (simulationManager != null)
            {
                simulationFrame = simulationManager.m_currentFrameIndex;
                gameTime = simulationManager.m_currentGameTime.ToString("o", CultureInfo.InvariantCulture);
            }

            for (int i = 0; i < s_vehicleIds.Count; i++)
            {
                SampleVehicle(vehicleManager, simulationFrame, gameTime, s_vehicleIds[i]);
            }

            if (s_linesSinceFlush >= FlushEveryLines)
            {
                FlushAllWriters();
                s_linesSinceFlush = 0;
            }
        }

        private static void SampleVehicle(
            VehicleManager vehicleManager,
            uint simulationFrame,
            string gameTime,
            ushort vehicleId)
        {
            StreamWriter writer;
            if (!s_writers.TryGetValue(vehicleId, out writer))
            {
                return;
            }

            if (vehicleManager == null)
            {
                return;
            }

            if (vehicleId >= VehicleManager.MAX_VEHICLE_COUNT)
            {
                return;
            }

            Vehicle vehicle = vehicleManager.m_vehicles.m_buffer[vehicleId];

            Vehicle.Flags flags = vehicle.m_flags;

            bool created = (flags & Vehicle.Flags.Created) != 0;
            bool deleted = (flags & Vehicle.Flags.Deleted) != 0;

            if (!created)
            {
                return;
            }

            if (deleted)
            {
                return;
            }

            Vehicle.Frame frame = vehicle.GetLastFrameData();

            Vector3 position = frame.m_position;
            Quaternion rotation = frame.m_rotation;
            Vector3 velocity = frame.m_velocity;

            string prefabName = "";

            try
            {
                VehicleInfo info = vehicle.Info;
                if (info != null)
                {
                    prefabName = info.name;
                }
            }
            catch
            {
                prefabName = "";
            }

            WriteCsv(
                writer,
                DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
                FormatFloat(s_sessionSeconds),
                simulationFrame.ToString(CultureInfo.InvariantCulture),
                gameTime,
                vehicleId.ToString(CultureInfo.InvariantCulture),
                FormatFloat(position.x),
                FormatFloat(position.y),
                FormatFloat(position.z),
                FormatFloat(rotation.eulerAngles.x),
                FormatFloat(rotation.eulerAngles.y),
                FormatFloat(velocity.magnitude),
                prefabName
            );
        }

        private static void FlushAllWriters()
        {
            foreach (KeyValuePair<ushort, StreamWriter> pair in s_writers)
            {
                pair.Value.Flush();
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
            s_linesSinceFlush++;
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
                @"Colossal Order\Cities_Skylines\Addons\Mods\VehicleTrajectoryRecorder"
            );
        }

        private static void Log(string message)
        {
            string full = "[VehicleTrajectoryRecorder] " + message;

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
