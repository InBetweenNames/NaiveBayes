// NaiveBayes.cpp : Defines the entry point for the console application.
//

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>


using PaperID = int64_t;
using Point = std::pair<PaperID, std::string>;
using ClassMetadata = std::pair<std::string, std::vector<Point>>;

template <typename T>
using Vec = std::vector<T, Eigen::aligned_allocator<T>>;

std::vector<Point> consume_metadata(const std::string& filename)
{
	std::vector<Point> points;

	std::ifstream metadata{ filename };


	//Read in the "reduced" title with no punctuation, all lowercase, etc


	std::string paper_orig_title;
	std::string paper_title;
	std::string paper_id_raw;
	while (std::getline(metadata, paper_id_raw, '\t'))
	{
		std::getline(metadata, paper_orig_title, '\t');
		std::getline(metadata, paper_title, '\t');

		PaperID paper_id = std::stoll(paper_id_raw, nullptr, 16);

		points.emplace_back(paper_id, paper_title);
		metadata.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

	return points;
}

//Extract all words from a vector of metadata and add them to a vocabulary set
void extract_vocabulary(const std::vector<Point>& metadata, std::set<std::string>& vocabulary)
{
	for (const auto& docs : metadata)
	{
		const auto& doc = docs.second;
		std::stringstream tokens{ doc };
		std::string token;
		while (tokens >> token)
		{
			vocabulary.emplace(token);
		}
	}
}

std::set<std::string> vocabulary(const std::vector<ClassMetadata>& metadata)
{
	std::set<std::string> vocabulary;

	for (size_t c = 0; c < metadata.size(); c++)
	{
		const auto& class_metadata = metadata[c].second;
		extract_vocabulary(class_metadata, vocabulary);
	}

	return vocabulary;

}

class MultinomialNaiveBayes
{
	int64_t N = 0;
	std::vector<std::string> V;
	Eigen::ArrayXd prior;
	Eigen::ArrayXXd condProb;
public:

	MultinomialNaiveBayes(const std::vector<ClassMetadata>& metadata) : MultinomialNaiveBayes{ metadata, vocabulary(metadata) } {}

	MultinomialNaiveBayes(const std::vector<ClassMetadata>& metadata, const std::set<std::string>& vocabulary)
	{

		//Initialize vocabulary (specific to this training set)

		V = std::vector<std::string>{ vocabulary.begin(), vocabulary.end() };


		const auto n_classes = metadata.size();
		Eigen::ArrayXd Nc{ n_classes };

		for (size_t i = 0; i < n_classes; i++)
		{
			Nc(i) = metadata[i].second.size();
			N += metadata[i].second.size();
		}

		//Now we have N, Nc, and V.  Priors are calculated for all classes here.
		prior = Nc / static_cast<double>(N);

		//Initialize conditional probabilities
		condProb = Eigen::ArrayXXd{ V.size(), n_classes };

		for (size_t i = 0; i < n_classes; i++)
		{
			std::map<std::string, int> occurrences;
			int64_t sumOccurrences = 0;

			for (const auto& point : metadata[i].second)
			{
				const auto& doc = point.second;
				std::stringstream tokens{ doc };
				std::string token;
				while (tokens >> token)
				{

					//TODO: for feature selection, will need to modify sumOccurrences here to only accumulate tokens that are in V
					++occurrences[token];
					++sumOccurrences;
				}
			}

			//Include laplace smoothing
			const auto denominator = sumOccurrences + V.size();

			Eigen::Index tIndex = 0;
			for (const auto& t : V)
			{
				int T_tc;
				try
				{
					T_tc = occurrences.at(t);
				}
				catch (const std::out_of_range&)
				{
					T_tc = 0;
				}

				//Include laplace smoothing
				const auto numerator = T_tc + 1;

				condProb(tIndex, i) = static_cast<double>(numerator) / static_cast<double>(denominator);

				tIndex++;
			}
		}

		//Store probabilities as logs in advance
		prior = prior.log();
		condProb = condProb.log();
	}

	Eigen::Index classify(const std::string& d)
	{
		std::stringstream tokens{ d };

		//TODO: do reductions to lowercase, punctuation, etc, in-program instead, and do it here as well as in the training
		std::string token;
		Eigen::ArrayXd scores = prior;

		//Logs already taken
		while (tokens >> token)
		{
			const auto& res = std::lower_bound(V.cbegin(), V.cend(), token);
			if (res == V.cend())
			{
				//TODO: if word is not in vocabulary, then compute probability accordingly (currently the word is ignored)
				/*for (Eigen::Index c = 0; c < prior.size(); c++)
				{
					scores(c) += std::log(1 / V.size());
				}*/
				continue;
			}
			else
			{
				const auto t = std::distance(V.cbegin(), res);

				for (Eigen::Index c = 0; c < prior.size(); c++)
				{
					scores(c) += condProb(t, c);
				}
			}
		}
		Eigen::Index winner;
		scores.maxCoeff(&winner);

		return winner;
	}

};

Eigen::Array<Eigen::Array2i, -1, -1> get_m_fold_slices(const std::vector<ClassMetadata>& metadata, int M)
{
	const auto n_classes = metadata.size();
	Eigen::Array<Eigen::Array2i, -1, -1> indices{M, n_classes};

	for (int i = 0; i < n_classes; i++)
	{
		const auto Nc = metadata[i].second.size();
		const auto n_per_fold = Nc / M;
		//const auto n_per_fold_r = Nc % M;

		for (int j = 0; j < M; j++)
		{
			const auto start = j*n_per_fold;


			indices(j, i)(0) = static_cast<int>(start);
			
			//Take care to include remainder in the last fold
			if (j == M - 1)
			{
				const auto end = Nc;
				indices(j, i)(1) = static_cast<int>(end);
			}
			else
			{
				const auto end = (j + 1)*n_per_fold;
				indices(j, i)(1) = static_cast<int>(end);
			}
		}
	}

	return indices;
}

Vec<Vec<Eigen::Array2i>> get_m_fold_training_indices(const Eigen::Array<Eigen::Array2i, -1, -1>& indices, const Eigen::Index exclude)
{
	Vec<Vec<Eigen::Array2i>> classChunks(indices.cols());

	for (Eigen::Index c = 0; c < indices.cols(); c++)
	{
		auto& chunks = classChunks[c];

		if (exclude != 0)
		{
			Eigen::Vector2i chunk = { indices(0, c)(0), indices(exclude - 1, c)(1) };
			chunks.emplace_back(chunk);
		}

		if (exclude != indices.rows() - 1)
		{
			Eigen::Vector2i chunk = { indices(exclude + 1, c)(0), indices(indices.rows() - 1, c)(1) };
			chunks.emplace_back(chunk);
		}
	}

	return classChunks;
}

Eigen::Array<Eigen::Array2i, 1, -1> get_m_fold_testing_indices(const Eigen::Array<Eigen::Array2i, -1, -1>& indices, const Eigen::Index include)
{
	return indices.row(include);
}

int __cdecl main(int argc, char* argv[])
{
	std::vector<std::pair<std::string, std::string>> metadata_files;
	int n_features = 0;
	if (argc < 2)
	{
		metadata_files.emplace_back("icse", "metadata/icse_id.txt");
		metadata_files.emplace_back("vldb", "metadata/vldb_id.txt");
	}
	else
	{
		std::string first_arg{ argv[1] };
		int start_index = 1;
		if (first_arg == "--selectfeatures")
		{
			if (argc < 3)
			{
				n_features = 100; //Default to selecting 100 features
				start_index = 2;
			}
			else
			{
				try
				{
					n_features = std::stoi(argv[2]);
					start_index = 3;
					if (n_features < 0)
					{
						n_features = 100;
					}
				}
				catch (...)
				{
					start_index = 2;
				}
			}
		}
		if (first_arg == "--help" || (argc - start_index) % 2 != 0)
		{
			std::cout << "Usage: " << argv[0] << " (--selectfeatures (<n>)) [class1 class1filename class2 class2filename ... classN classNfilename]" << std::endl;
			std::cout << "If no arguments are provided, this command line will be run:\n\t" << argv[0] << " --selectfeatures 100 icse metadata/icse_id.txt vldb metadata/vldb_id.txt" << std::endl;

			return 0;
		}
		for (int i = start_index; i < argc; i += 2)
		{
			const auto* class_name = argv[i];
			const auto* filename = argv[i + 1];
			metadata_files.emplace_back(class_name, filename);
		}

	}

	std::vector<ClassMetadata> metadata;


	//Load in metadata from each file
	const Eigen::Index n_classes = static_cast<Eigen::Index>(metadata_files.size());
	int c = 0;
	for (const auto& file : metadata_files)
	{
		const auto& class_metadata = consume_metadata(file.second);
		metadata.emplace_back(file.first, class_metadata);
		std::cout << "Loading " << file.first << " (class " << c << ") " << file.second << std::endl;
		c++;
	}

	const auto M = 10;
	std::cout << "Begin " << M << "-fold cross validation" << std::endl;


	const auto folds = get_m_fold_slices(metadata, M);


	Eigen::ArrayXXi confusionMatrix = Eigen::ArrayXXi::Zero(n_classes, n_classes);
	//MultinomialNaiveBayes classifier{ metadata, V };
	for (Eigen::Index i = 0; i < M; i++)
	{
		std::cout << "Fold " << (i + 1) << std::endl;
		const auto trainingIndices = get_m_fold_training_indices(folds, i);
		const auto testIndices = get_m_fold_testing_indices(folds, i);

		std::vector<ClassMetadata> test_metadata;
		for (size_t c1 = 0; c1 < metadata.size(); c1++)
		{
			std::vector<Point> trainingPoints;
			for (const auto& range : trainingIndices[c1])
			for (size_t p1 = range(0); p1 < range(1); p1++)
			{
				trainingPoints.emplace_back(metadata[c1].second[p1]);
			}
			test_metadata.emplace_back(metadata[c1].first, trainingPoints);
		}

		MultinomialNaiveBayes classifier{ test_metadata };

		for (Eigen::Index c = 0; c < n_classes; c++)
		{
			const Eigen::Array2i range = testIndices(0, c);
			//std::cout << "Testing range:\n" << range << std::endl;
			/*std::cout << "Training ranges:\n";
			for (const auto& C : trainingIndices)
			{
				std::cout << "class" << std::endl;
				for (const auto& range : C)
				{
					std::cout << range << std::endl;
				}
			}*/
			
			for (Eigen::Index testDoc = range(0); testDoc < range(1); testDoc++)
			{
				const auto& doc = metadata[c].second[testDoc];
				const auto& docTitle = doc.second; //TODO: perform reductions in here as well as in training

				const auto resultClass = classifier.classify(docTitle);

				confusionMatrix(c, resultClass)++;
			}
		}

	}

	std::cout << "Final confusion matrix:\n" << confusionMatrix << std::endl;

	double tp = confusionMatrix(0, 0);
	double fn = confusionMatrix(0, 1);
	double fp = confusionMatrix(1, 0);
	double tn = confusionMatrix(1, 1);

	double precision = tp / (tp + fp);
	double recall = tp / (tp + fn);
	double F1 = 2 * precision * recall / (precision + recall);
	std::cout << "Precision: " << precision << std::endl;
	std::cout << "Recall: " << recall << std::endl;
	std::cout << "F1 measure: " << F1 << std::endl;

	//m_fold_cross_validate(metadata, V)

	//MultinomialNaiveBayes all{ metadata , V };
	//std::string test = "logic programming environments for large knowledge bases a practical perspective abstract";
	//std::cout << test << ": " << all.classify(test);




    return 0;
}

