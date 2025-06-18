
import 'package:flutter/material.dart';
import 'package:front_electricity_forecasting/models/features.dart';
import 'package:front_electricity_forecasting/models/predict.dart';
import 'package:http/http.dart' as http;

class PredictsProvider extends ChangeNotifier {

  PredictsProvider() {
    // fetchFeatures('2023-07-08');
  }

  final String baseUrl = 'http://127.0.0.1:8000/api/v1';

  String _currentDate = '';
  FeatureResponse? features;
  PredictResponse? prediction;

  String get currentDate => _currentDate;
  set currentDate(String value) {
    _currentDate = value;
    notifyListeners();
  }
  
  bool _isLoading = false;
  bool get isLoading => _isLoading;
  set isLoading(bool value) {
    _isLoading = value;
    notifyListeners();
  }

  Future<bool> fetch(String apikey) async {
    isLoading = true;
    await fetchPrediction(apikey);
    if (prediction != null) {
      print('Prediction fetched for date: ${prediction!.date}');
      await fetchFeatures(apikey, currentDate);
      notifyListeners();
      return true;
    }
    return false;
  }

  Future<void> fetchFeatures(String apiKey, String date) async {
    isLoading = true;
    final Uri url = Uri.parse('$baseUrl/features?date=$date');

    try {
      final response = await http.get(url, headers: {'Authorization': apiKey});
      if (response.statusCode == 200) {
        features = FeatureResponse.fromJson(response.body);
      } else {
        print('Error: ${response.statusCode}, ${response.body}');
      }
    } catch (e) {
      print('Error fetching features: $e');
    } finally {
      isLoading = false;
    }
  }

  Future<void> fetchPrediction(String apiKey) async {
    isLoading = true;
    final Uri url = Uri.parse('$baseUrl/predict');

    try {
      final response = await http.get(url, headers: {'Authorization': apiKey});

      if (response.statusCode == 200) {
        prediction = PredictResponse.fromJson(response.body);
        currentDate = prediction!.date;
      } else {
        print('Error: ${response.statusCode}, ${response.body}');
      }
    } catch (e) {
      print('Error fetching Prediction: $e');
    } finally {
      isLoading = false;
    }
  }

}