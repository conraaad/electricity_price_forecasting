import 'dart:convert';

class PredictResponse {
    String nameModel;
    String date;
    DailyMean dailyMean;
    Map<String, HourPrediction> hourPredictions;

    PredictResponse({
        required this.nameModel,
        required this.date,
        required this.dailyMean,
        required this.hourPredictions,
    });

    factory PredictResponse.fromJson(String str) => PredictResponse.fromMap(json.decode(str));

    String toJson() => json.encode(toMap());

    factory PredictResponse.fromMap(Map<String, dynamic> json) => PredictResponse(
        nameModel: json["name_model"],
        date: json["date"],
        dailyMean: DailyMean.fromMap(json["daily_mean"]),
        hourPredictions: Map.from(json["hour_predictions"]).map((k, v) => MapEntry<String, HourPrediction>(k, HourPrediction.fromMap(v))),
    );

    Map<String, dynamic> toMap() => {
        "name_model": nameModel,
        "date": date,
        "daily_mean": dailyMean.toMap(),
        "hour_predictions": Map.from(hourPredictions).map((k, v) => MapEntry<String, dynamic>(k, v.toMap())),
    };
}

class DailyMean {
    double mae;
    double rmse;
    double smape;

    DailyMean({
        required this.mae,
        required this.rmse,
        required this.smape,
    });

    factory DailyMean.fromJson(String str) => DailyMean.fromMap(json.decode(str));

    String toJson() => json.encode(toMap());

    factory DailyMean.fromMap(Map<String, dynamic> json) => DailyMean(
        mae: json["mae"]?.toDouble(),
        rmse: json["rmse"]?.toDouble(),
        smape: json["smape"]?.toDouble(),
    );

    Map<String, dynamic> toMap() => {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
    };
}

class HourPrediction {
    double mae;
    double rmse;
    double smape;
    double predictedPrice;

    HourPrediction({
        required this.mae,
        required this.rmse,
        required this.smape,
        required this.predictedPrice,
    });

    factory HourPrediction.fromJson(String str) => HourPrediction.fromMap(json.decode(str));

    String toJson() => json.encode(toMap());

    factory HourPrediction.fromMap(Map<String, dynamic> json) => HourPrediction(
        mae: json["mae"]?.toDouble(),
        rmse: json["rmse"]?.toDouble(),
        smape: json["smape"]?.toDouble(),
        predictedPrice: json["predicted_price"]?.toDouble(),
    );

    Map<String, dynamic> toMap() => {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "predicted_price": predictedPrice,
    };
}
