using System.Runtime.InteropServices;
using CSnakes.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using TransformersSharp.Pipelines;

namespace TransformersSharp
{
    public static class TransformerEnvironment
    {
        private static readonly IPythonEnvironment? _env;
        private static readonly Lock _setupLock = new();

        static TransformerEnvironment()
        {
            lock (_setupLock)
            {
                IHostBuilder builder = Host.CreateDefaultBuilder()
                    .ConfigureServices(services =>
                    {
                        // Use Local AppData folder for Python installation
                        string appDataPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TransformersSharp");

                        // Create the directory if it doesn't exist
                        if (!Directory.Exists(appDataPath))
                            Directory.CreateDirectory(appDataPath);

                        // If user has an environment variable TRANSFORMERS_SHARP_VENV_PATH, use that instead
                        string? envPath = Environment.GetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH");
                        string venvPath;
                        if (envPath != null)
                            venvPath = envPath;
                        else
                            venvPath = Path.Join(appDataPath, "venv");

                        // Write requirements to appDataPath
                        string requirementsPath = Path.Join(appDataPath, "requirements.txt");

                        // TODO: Make this configurable
                        string[] requirements =
                        {
                            "--extra-index-url https://download.pytorch.org/whl/nightly/cpu",
                            "torch==2.10.0.dev20250918",
                            "transformers",
                            "tokenizers @ https://files.pythonhosted.org/packages/5e/b4/c1ce3699e81977da2ace8b16d2badfd42b060e7d33d75c4ccdbf9dc920fa/tokenizers-0.22.0.tar.gz",
                            "pyyaml @ https://files.pythonhosted.org/packages/54/ed/79a089b6be93607fa5cdaedf301d7dfb23af5f25c398d5ead2525b063e17/pyyaml-6.0.2.tar.gz",
                            "safetensors @ https://files.pythonhosted.org/packages/ac/cc/738f3011628920e027a11754d9cae9abec1aed00f7ae860abbf843755233/safetensors-0.6.2.tar.gz",
                            "sentence_transformers",
                            "scikit-learn @ git+https://github.com/vask2108/scikit-learn@woa_gha_runner",
                            "pillow",
                            "timm",
                            "einops"
                        };

                        File.WriteAllText(requirementsPath, string.Join('\n', requirements));

                        services
                                .WithPython()
                                .WithHome(appDataPath)
                                .WithVirtualEnvironment(venvPath)
                                .WithUvInstaller()
                                .FromRedistributable(); // Download Python 3.12 and store it locally
                    });

                var app = builder.Build();

                _env = app.Services.GetRequiredService<IPythonEnvironment>();
            }
        }

        private static IPythonEnvironment Env => _env ?? throw new InvalidOperationException("Python environment is not initialized..");

        internal static ITransformersWrapper TransformersWrapper => Env.TransformersWrapper();
        internal static ISentenceTransformersWrapper SentenceTransformersWrapper => Env.SentenceTransformersWrapper();

        /// <summary>
        /// Login to Huggingface with a token.
        /// </summary>
        /// <param name="token"></param>
        public static void Login(string token)
        {
            var wrapperModule = Env.TransformersWrapper();
            wrapperModule.HuggingfaceLogin(token);
        }

        public static Pipeline Pipeline(string? task = null, string? model = null, string? tokenizer = null, TorchDtype? torchDtype = null)
        {
            var wrapperModule = Env.TransformersWrapper();
            string? torchDtypeStr = torchDtype?.ToString() ?? null;
            var pipeline = wrapperModule.Pipeline(task, model, tokenizer, torchDtypeStr);

            return new Pipeline(pipeline);
        }

        public static void Dispose()
        {
            _env?.Dispose();
        }
    }
}
