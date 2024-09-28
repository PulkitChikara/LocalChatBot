using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Forms;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using UglyToad.PdfPig;
using Microsoft.ML;
using System.Data;
using System.Windows.Threading;

namespace LocalChatBot
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        ITransformer trainModel = null;
        bool isTraining = false;

        public MainWindow()
        {
            InitializeComponent();
        }

        public string ExtractTextFromPDF(string filePath)
        {
            using (var document = PdfDocument.Open(filePath))
            {
                string extractedText = "";
                foreach (var page in document.GetPages())
                {
                    extractedText += page.Text;
                }
                return extractedText;
            }
        }

        public List<ModelInput> ExtractTextFromPDFFolder(string folderPath)
        {
            var pdfFiles = Directory.GetFiles(folderPath, "*.pdf");
            var modelInputList = new List<ModelInput>();

            foreach (var file in pdfFiles)
            {
                string text = ExtractTextFromPDF(file);
                //text = text.ToLower();
                //text = string.Concat(text.Where(c => !char.IsPunctuation(c)));

                ModelInput modelInput = new ModelInput();
                modelInput.Text = text;
                modelInputList.Add(modelInput);
            }

            return modelInputList;
        }

        public async void TrainModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (isTraining) return;

            using (var folderDialog = new FolderBrowserDialog())
            {
                folderDialog.Description = "Select a folder";
                folderDialog.ShowNewFolderButton = true;

                // Show the FolderBrowserDialog
                var result = folderDialog.ShowDialog();

                // If the user selected a folder, display it in the TextBox
                if (result == System.Windows.Forms.DialogResult.OK && !string.IsNullOrWhiteSpace(folderDialog.SelectedPath))
                {
                    DocumentsPath.Text = folderDialog.SelectedPath;

                    Dispatcher.Invoke(() =>
                    {
                        TrainingLabel.Visibility = Visibility.Visible;
                    });

                    List<ModelInput> modelInputList = ExtractTextFromPDFFolder(DocumentsPath.Text);
                    if (modelInputList.Count == 0)
                    {
                        System.Windows.MessageBox.Show("Please select the folder including pdf files.");
                        return;
                    }

                    isTraining = true;

                    await Task.Run(() =>
                    {
                        trainModel = MLModel.TrainModel(modelInputList);

                        Dispatcher.Invoke(() =>
                        {
                            TrainingLabel.Visibility = Visibility.Hidden;
                            System.Windows.MessageBox.Show("Training is finished!");
                        });

                        isTraining = false;
                    });
                }
            }            
        }

        void AskButton_Click(object sender, RoutedEventArgs e)
        {
            if (isTraining) return;

            if (trainModel == null)
            {
                System.Windows.MessageBox.Show("Please train model with pdf files.");
                return;
            }

            string userInput = UserInput.Text.Trim();

            if (string.IsNullOrEmpty(userInput))
            {
                System.Windows.MessageBox.Show("Please enter a query.");
                return;
            }

            ModelOutput output = MLModel.Predict(trainModel, userInput);

            ChatTextBox.Text += $"User: {userInput}\nBot: {output.Prediction}\n";
            UserInput.Clear();
        }
    }
}
